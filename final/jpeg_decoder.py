# jpeg_decoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

# ZigZag order index -> (row, col) in 8x8
ZIGZAG_POS = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

def _read_u16(be_bytes: bytes, i: int) -> Tuple[int, int]:
    return (be_bytes[i] << 8) | be_bytes[i+1], i + 2

class BitReader:
    """
    Reads bits from entropy-coded segment. Handles byte-stuffing (0xFF 0x00).
    Stops before markers (0xFF followed by non-0x00).
    """
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0
        self.eoi = False

    def _fill(self):
        while self.bit_cnt <= 16 and self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            if b == 0xFF:
                if self.pos >= len(self.data):
                    self.eoi = True
                    return
                nxt = self.data[self.pos]
                # stuffed 0x00 means literal 0xFF
                if nxt == 0x00:
                    self.pos += 1
                    b = 0xFF
                else:
                    # marker encountered; step back so caller can handle
                    self.pos -= 1  # keep at 0xFF
                    self.eoi = True
                    return
            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_cnt += 8

    def get_bits(self, n: int) -> int:
        if n == 0:
            return 0
        while self.bit_cnt < n and not self.eoi:
            self._fill()
        if self.bit_cnt < n:
            # stream ended unexpectedly
            raise EOFError("Not enough bits in entropy stream")
        shift = self.bit_cnt - n
        val = (self.bit_buf >> shift) & ((1 << n) - 1)
        self.bit_cnt -= n
        self.bit_buf &= (1 << self.bit_cnt) - 1 if self.bit_cnt > 0 else 0
        return val

    def get_bit(self) -> int:
        return self.get_bits(1)

@dataclass
class HuffmanTable:
    # maps (length, code) -> symbol
    codes: Dict[Tuple[int, int], int]
    max_len: int

    def decode(self, br: BitReader) -> int:
        code = 0
        for length in range(1, self.max_len + 1):
            code = (code << 1) | br.get_bit()
            key = (length, code)
            if key in self.codes:
                return self.codes[key]
        raise ValueError("Huffman decode failed")

def build_huffman_table(lengths: List[int], symbols: List[int]) -> HuffmanTable:
    """
    JPEG DHT gives 16 counts for code lengths 1..16, then symbols in order.
    Build canonical Huffman codes.
    """
    codes: Dict[Tuple[int, int], int] = {}
    code = 0
    k = 0
    max_len = 0
    for length in range(1, 17):
        count = lengths[length - 1]
        if count:
            max_len = max(max_len, length)
        for _ in range(count):
            sym = symbols[k]
            codes[(length, code)] = sym
            code += 1
            k += 1
        code <<= 1
    return HuffmanTable(codes=codes, max_len=max_len if max_len else 16)

def receive_extend(v: int, t: int) -> int:
    """
    Convert additional bits (v) with category t into signed value.
    JPEG sign extension rule.
    """
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - ((1 << t) - 1)
    return v

@dataclass
class Component:
    cid: int
    h: int
    v: int
    tq: int  # quant table id

@dataclass
class SOSComponentSpec:
    cid: int
    td: int  # DC table id
    ta: int  # AC table id

class JPEGParser:
    def __init__(self, jpeg_bytes: bytes):
        self.b = jpeg_bytes
        self.qtables: Dict[int, np.ndarray] = {}
        self.ht_dc: Dict[int, HuffmanTable] = {}
        self.ht_ac: Dict[int, HuffmanTable] = {}
        self.width = 0
        self.height = 0
        self.components: Dict[int, Component] = {}
        self.sos_specs: List[SOSComponentSpec] = []
        self.entropy_data: bytes = b""

    def parse(self):
        b = self.b
        i = 0
        if b[0:2] != b"\xFF\xD8":
            raise ValueError("Not a JPEG (missing SOI)")
        i = 2

        def read_marker():
            nonlocal i
            # find 0xFF
            while i < len(b) and b[i] != 0xFF:
                i += 1
            if i >= len(b):
                return None
            while i < len(b) and b[i] == 0xFF:
                i += 1
            if i >= len(b):
                return None
            m = b[i]
            i += 1
            return m

        while True:
            m = read_marker()
            if m is None:
                break
            if m == 0xD9:  # EOI
                break
            if m in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7):  # RSTn
                continue
            if m == 0xDA:  # SOS
                seglen, i = _read_u16(b, i)
                seg = b[i:i+seglen-2]
                i += seglen - 2
                self._parse_sos(seg)
                # entropy data until EOI (or next marker)
                self.entropy_data = b[i:]
                break

            # segments with length
            seglen, i = _read_u16(b, i)
            seg = b[i:i+seglen-2]
            i += seglen - 2

            if m == 0xDB:  # DQT
                self._parse_dqt(seg)
            elif m == 0xC4:  # DHT
                self._parse_dht(seg)
            elif m == 0xC0:  # SOF0 (baseline)
                self._parse_sof0(seg)
            else:
                # ignore APPx, COM, etc.
                pass

        if not self.entropy_data:
            raise ValueError("Missing SOS/entropy data")

    def _parse_dqt(self, seg: bytes):
        i = 0
        while i < len(seg):
            pq_tq = seg[i]
            i += 1
            pq = (pq_tq >> 4) & 0x0F
            tq = pq_tq & 0x0F
            if pq != 0:
                raise NotImplementedError("16-bit quant table not supported")
            qt = np.frombuffer(seg[i:i+64], dtype=np.uint8).astype(np.float32)
            i += 64
            # quant table is in zigzag order in file; convert to 8x8 natural order
            q8 = np.zeros((8,8), dtype=np.float32)
            for k,(r,c) in enumerate(ZIGZAG_POS):
                q8[r,c] = qt[k]
            self.qtables[tq] = q8

    def _parse_dht(self, seg: bytes):
        i = 0
        while i < len(seg):
            tc_th = seg[i]; i += 1
            tc = (tc_th >> 4) & 0x0F  # 0=DC, 1=AC
            th = tc_th & 0x0F
            lengths = list(seg[i:i+16]); i += 16
            total = sum(lengths)
            symbols = list(seg[i:i+total]); i += total
            ht = build_huffman_table(lengths, symbols)
            if tc == 0:
                self.ht_dc[th] = ht
            else:
                self.ht_ac[th] = ht

    def _parse_sof0(self, seg: bytes):
        # [P][Y][X][Nf][comp...]
        p = seg[0]
        if p != 8:
            raise NotImplementedError("Only 8-bit precision supported")
        self.height = (seg[1] << 8) | seg[2]
        self.width  = (seg[3] << 8) | seg[4]
        nf = seg[5]
        i = 6
        self.components = {}
        for _ in range(nf):
            cid = seg[i]; hv = seg[i+1]; tq = seg[i+2]
            i += 3
            h = (hv >> 4) & 0x0F
            v = hv & 0x0F
            self.components[cid] = Component(cid=cid, h=h, v=v, tq=tq)

    def _parse_sos(self, seg: bytes):
        ns = seg[0]
        i = 1
        specs = []
        for _ in range(ns):
            cid = seg[i]; tdta = seg[i+1]; i += 2
            td = (tdta >> 4) & 0x0F
            ta = tdta & 0x0F
            specs.append(SOSComponentSpec(cid=cid, td=td, ta=ta))
        self.sos_specs = specs
        # ignore Ss, Se, Ah/Al (baseline should be 0,63,0)

def decode_one_block(br: BitReader, ht_dc: HuffmanTable, ht_ac: HuffmanTable,
                     prev_dc: int, zigzag_on: bool) -> Tuple[np.ndarray, int]:
    """
    Return: (coeff8x8 in natural order), updated_prev_dc
    """
    coeff = np.zeros((8,8), dtype=np.int16)

    # DC
    s = ht_dc.decode(br)  # category
    v = br.get_bits(s)
    diff = receive_extend(v, s)
    dc = prev_dc + diff
    prev_dc = dc
    if zigzag_on:
        r,c = ZIGZAG_POS[0]
        coeff[r,c] = dc
    else:
        coeff[0,0] = dc

    # AC
    k = 1
    while k < 64:
        rs = ht_ac.decode(br)
        if rs == 0x00:  # EOB
            break
        if rs == 0xF0:  # ZRL
            run = 16
            size = 0
        else:
            run = (rs >> 4) & 0x0F
            size = rs & 0x0F

        k += run
        if k >= 64:
            break
        if size > 0:
            v = br.get_bits(size)
            ac = receive_extend(v, size)
        else:
            ac = 0

        if zigzag_on:
            r,c = ZIGZAG_POS[k]
            coeff[r,c] = ac
        else:
            # "zigzag off": fill in raster order index k
            rr = k // 8
            cc = k % 8
            coeff[rr,cc] = ac

        k += 1

    return coeff.astype(np.float32), prev_dc

def decode_baseline_huffman(jpeg_bytes: bytes,
                           zigzag_on: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Decode baseline JPEG entropy stream into quantized DCT blocks, then return per-channel
    coefficient blocks (not yet dequant/IDCT):
      Y_blocks: (bh, bw, 8, 8)
      Cb_blocks: ...
      Cr_blocks: ...
    Assumes 3 components present in one scan.
    For course-project simplicity: expects no subsampling (all H=V=1).
    """
    jp = JPEGParser(jpeg_bytes)
    jp.parse()

    # Expect 3 components: typical ids 1(Y),2(Cb),3(Cr)
    # If your files differ, we still handle by CID mapping in SOS order.
    comps_in_scan = jp.sos_specs
    if len(comps_in_scan) != 3:
        raise NotImplementedError("This decoder expects 3-component JPEG (YCbCr)")

    # For simplicity, require 4:4:4 (h=v=1 for all)
    for spec in comps_in_scan:
        c = jp.components[spec.cid]
        if not (c.h == 1 and c.v == 1):
            raise NotImplementedError(
                "Subsampling not supported in this baseline decoder. "
                "Please encode JPG with subsampling=0 (4:4:4)."
            )

    H, W = jp.height, jp.width
    bw = (W + 7) // 8
    bh = (H + 7) // 8

    # allocate blocks
    blocks = {spec.cid: np.zeros((bh, bw, 8, 8), dtype=np.float32) for spec in comps_in_scan}
    prev_dc = {spec.cid: 0 for spec in comps_in_scan}

    # bit reader on entropy data; it may include EOI marker; BitReader stops at marker
    br = BitReader(jp.entropy_data)

    for by in range(bh):
        for bx in range(bw):
            for spec in comps_in_scan:
                htD = jp.ht_dc[spec.td]
                htA = jp.ht_ac[spec.ta]
                coeff8, prev = decode_one_block(br, htD, htA, prev_dc[spec.cid], zigzag_on)
                prev_dc[spec.cid] = prev
                blocks[spec.cid][by, bx] = coeff8

    meta = {
        "width": W, "height": H,
        "qtables": jp.qtables,
        "components": jp.components,
        "sos_specs": jp.sos_specs,
    }
    # return in SOS order as Y,Cb,Cr planes (by cid)
    # Most JPEGs use cid 1,2,3. If not, we map by SOS order:
    cid_Y, cid_Cb, cid_Cr = [s.cid for s in comps_in_scan]
    return blocks[cid_Y], blocks[cid_Cb], blocks[cid_Cr], meta
