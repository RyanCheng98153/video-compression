# jpeg_decoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

# Standard JPEG zigzag positions: index -> (r,c)
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

def _read_u16(b: bytes, i: int) -> Tuple[int, int]:
    return (b[i] << 8) | b[i+1], i + 2


# ---------------------------
# Huffman
# ---------------------------
@dataclass
class HuffmanTable:
    # (length, code) -> symbol
    codes: Dict[Tuple[int, int], int]
    max_len: int

    def decode(self, br: "EntropyBitReader") -> int:
        code = 0
        for length in range(1, self.max_len + 1):
            code = (code << 1) | br.get_bit()
            sym = self.codes.get((length, code))
            if sym is not None:
                return sym
        raise ValueError("Huffman decode failed")


def build_huffman_table(lengths: List[int], symbols: List[int]) -> HuffmanTable:
    """
    JPEG canonical Huffman code construction.
    lengths: 16 counts for lengths 1..16
    symbols: in order
    """
    codes: Dict[Tuple[int, int], int] = {}
    code = 0
    k = 0
    max_len = 0
    for length in range(1, 17):
        cnt = lengths[length - 1]
        if cnt:
            max_len = max(max_len, length)
        for _ in range(cnt):
            codes[(length, code)] = symbols[k]
            code += 1
            k += 1
        code <<= 1
    return HuffmanTable(codes=codes, max_len=max_len if max_len else 16)


def receive_extend(v: int, t: int) -> int:
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - ((1 << t) - 1)
    return v


# ---------------------------
# JPEG structures
# ---------------------------
@dataclass
class Component:
    cid: int
    h: int
    v: int
    tq: int


@dataclass
class SOSComponentSpec:
    cid: int
    td: int
    ta: int


# ---------------------------
# Entropy bit reader (scan data only)
# ---------------------------
class EntropyBitReader:
    """
    Read bits from scan entropy data, handling:
      - byte-stuffing: FF 00 -> data byte FF
      - restart markers: FF D0..D7 (set pending_rst and flush bit buffer)
    The input scan_data MUST still contain FF 00 pairs (do not unstuff earlier).
    """
    def __init__(self, scan_data: bytes):
        self.data = scan_data
        self.pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0
        self.pending_rst: Optional[int] = None
        self.done = False

    def reset_bits(self):
        self.bit_buf = 0
        self.bit_cnt = 0

    def _read_byte(self) -> Optional[int]:
        if self.pos >= len(self.data):
            return None
        v = self.data[self.pos]
        self.pos += 1
        return v

    def _fill(self):
        while self.bit_cnt <= 16 and not self.done and self.pending_rst is None:
            b = self._read_byte()
            if b is None:
                self.done = True
                return

            if b == 0xFF:
                nxt = self._read_byte()
                if nxt is None:
                    self.done = True
                    return

                if nxt == 0x00:
                    # stuffed FF -> literal data byte 0xFF
                    b = 0xFF
                elif 0xD0 <= nxt <= 0xD7:
                    # restart marker
                    self.pending_rst = nxt
                    self.reset_bits()
                    return
                else:
                    # Any other marker should NOT appear inside extracted scan bytes.
                    self.done = True
                    return

            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_cnt += 8

    def get_bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pending_rst is not None:
            raise RuntimeError("RST pending")

        while self.bit_cnt < n and not self.done:
            self._fill()
            if self.pending_rst is not None:
                raise RuntimeError("RST pending")

        if self.bit_cnt < n:
            raise EOFError("Not enough bits in scan data")

        shift = self.bit_cnt - n
        val = (self.bit_buf >> shift) & ((1 << n) - 1)
        self.bit_cnt -= n
        self.bit_buf &= (1 << self.bit_cnt) - 1 if self.bit_cnt > 0 else 0
        return val

    def get_bit(self) -> int:
        return self.get_bits(1)


# ---------------------------
# JPEG marker parser + scan extractor
# ---------------------------
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
        self.restart_interval = 0  # DRI (in MCUs)
        self.scan_data: bytes = b""

    def parse(self):
        b = self.b
        if b[:2] != b"\xFF\xD8":
            raise ValueError("Not a JPEG (missing SOI)")

        i = 2

        def read_marker():
            nonlocal i
            # seek 0xFF
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

            if m == 0xDA:  # SOS
                seglen, i = _read_u16(b, i)
                seg = b[i:i + seglen - 2]
                i += seglen - 2
                self._parse_sos(seg)
                self.scan_data, i = self._extract_scan_data(b, i)
                break

            if 0xD0 <= m <= 0xD7:
                # standalone RST (rare outside scan)
                continue

            seglen, i = _read_u16(b, i)
            seg = b[i:i + seglen - 2]
            i += seglen - 2

            if m == 0xDB:      # DQT
                self._parse_dqt(seg)
            elif m == 0xC4:    # DHT
                self._parse_dht(seg)
            elif m == 0xC0:    # SOF0
                self._parse_sof0(seg)
            elif m == 0xDD:    # DRI
                self._parse_dri(seg)
            else:
                # ignore APPx/COM/etc
                pass

        if not self.scan_data:
            raise ValueError("Missing SOS/scan data")

    def _parse_dri(self, seg: bytes):
        self.restart_interval = (seg[0] << 8) | seg[1]

    def _parse_dqt(self, seg: bytes):
        j = 0
        while j < len(seg):
            pq_tq = seg[j]; j += 1
            pq = (pq_tq >> 4) & 0x0F
            tq = pq_tq & 0x0F
            if pq != 0:
                raise NotImplementedError("16-bit DQT not supported")

            qt = np.frombuffer(seg[j:j+64], dtype=np.uint8).astype(np.float32)
            j += 64

            # JPEG stores table in zigzag order -> convert to natural 8x8
            q8 = np.zeros((8,8), dtype=np.float32)
            for k, (r,c) in enumerate(ZIGZAG_POS):
                q8[r,c] = qt[k]
            self.qtables[tq] = q8

    def _parse_dht(self, seg: bytes):
        j = 0
        while j < len(seg):
            tc_th = seg[j]; j += 1
            tc = (tc_th >> 4) & 0x0F  # 0 DC, 1 AC
            th = tc_th & 0x0F
            lengths = list(seg[j:j+16]); j += 16
            total = sum(lengths)
            symbols = list(seg[j:j+total]); j += total

            ht = build_huffman_table(lengths, symbols)
            if tc == 0:
                self.ht_dc[th] = ht
            else:
                self.ht_ac[th] = ht

    def _parse_sof0(self, seg: bytes):
        p = seg[0]
        if p != 8:
            raise NotImplementedError("Only 8-bit SOF0 supported")
        self.height = (seg[1] << 8) | seg[2]
        self.width  = (seg[3] << 8) | seg[4]
        nf = seg[5]
        j = 6
        self.components = {}
        for _ in range(nf):
            cid = seg[j]
            hv  = seg[j+1]
            tq  = seg[j+2]
            j += 3
            h = (hv >> 4) & 0x0F
            v = hv & 0x0F
            self.components[cid] = Component(cid=cid, h=h, v=v, tq=tq)

    def _parse_sos(self, seg: bytes):
        ns = seg[0]
        j = 1
        specs: List[SOSComponentSpec] = []
        for _ in range(ns):
            cid = seg[j]
            tdta = seg[j+1]
            j += 2
            td = (tdta >> 4) & 0x0F
            ta = tdta & 0x0F
            specs.append(SOSComponentSpec(cid=cid, td=td, ta=ta))
        self.sos_specs = specs

    def _extract_scan_data(self, b: bytes, start: int) -> Tuple[bytes, int]:
        """
        Extract scan entropy bytes from after SOS until next marker.
        IMPORTANT: do NOT unstuff here. Keep FF00 as-is.
        Keep restart markers FFD0..FFD7 as-is.
        Stop on any other marker (e.g. EOI FFD9).
        """
        out = bytearray()
        i = start
        while i < len(b):
            x = b[i]
            if x != 0xFF:
                out.append(x)
                i += 1
                continue

            if i + 1 >= len(b):
                break
            y = b[i+1]

            if y == 0x00:
                # stuffed: keep FF 00
                out.append(0xFF)
                out.append(0x00)
                i += 2
                continue

            if 0xD0 <= y <= 0xD7:
                # restart marker inside scan: keep FF Dn
                out.append(0xFF)
                out.append(y)
                i += 2
                continue

            # other marker ends scan
            break

        return bytes(out), i


# ---------------------------
# Block decode (Huffman -> quantized coeff 8x8)
# ---------------------------
def decode_one_block(
    br: EntropyBitReader,
    ht_dc: HuffmanTable,
    ht_ac: HuffmanTable,
    prev_dc: int,
    zigzag_on: bool
) -> Tuple[np.ndarray, int]:

    coeff = np.zeros((8,8), dtype=np.int16)

    # DC
    s = ht_dc.decode(br)
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
            run, size = 16, 0
        else:
            run = (rs >> 4) & 0x0F
            size = rs & 0x0F

        k += run
        if k >= 64:
            break

        if size > 0:
            vv = br.get_bits(size)
            ac = receive_extend(vv, size)
        else:
            ac = 0

        if zigzag_on:
            r,c = ZIGZAG_POS[k]
            coeff[r,c] = ac
        else:
            rr = k // 8
            cc = k % 8
            coeff[rr,cc] = ac

        k += 1

    return coeff.astype(np.float32), prev_dc


def decode_baseline_huffman(jpeg_bytes: bytes, zigzag_on: bool = True):
    """
    Baseline JPEG (SOF0) Huffman entropy decode into quantized 8x8 DCT coeff blocks.
    Assumptions for your course project:
      - single scan with 3 components (YCbCr interleaved)
      - 4:4:4 (h=v=1 for all components) -> encode with subsampling=0
    Supports:
      - byte stuffing FF00
      - restart markers FFD0..FFD7 (even if DRI=0)
      - DRI parsing (restart_interval), but we primarily react to actual markers
    """
    jp = JPEGParser(jpeg_bytes)
    jp.parse()

    comps = jp.sos_specs
    if len(comps) != 3:
        raise NotImplementedError("Expect 3 components in one scan (YCbCr).")

    # enforce 4:4:4
    for s in comps:
        c = jp.components[s.cid]
        if not (c.h == 1 and c.v == 1):
            raise NotImplementedError("Subsampling not supported; encode JPG with subsampling=0 (4:4:4).")

    H, W = jp.height, jp.width
    bw = (W + 7) // 8
    bh = (H + 7) // 8

    blocks = {s.cid: np.zeros((bh, bw, 8, 8), dtype=np.float32) for s in comps}
    prev_dc = {s.cid: 0 for s in comps}

    br = EntropyBitReader(jp.scan_data)

    def reset_on_rst():
        # reset DC predictors for all components in scan
        for cid in prev_dc:
            prev_dc[cid] = 0
        br.pending_rst = None
        br.reset_bits()

    # Decode MCU-by-MCU (here: 1 Y, 1 Cb, 1 Cr per MCU for 4:4:4)
    for by in range(bh):
        for bx in range(bw):

            # If an RST marker is pending before starting MCU, reset.
            if br.pending_rst is not None:
                reset_on_rst()

            for s in comps:
                # If an RST marker appears between blocks, reset and restart this MCU cleanly.
                if br.pending_rst is not None:
                    reset_on_rst()
                    # restart this MCU from scratch: clear what we wrote for this MCU
                    for ss in comps:
                        blocks[ss.cid][by, bx].fill(0)
                    # restart the per-component loop
                    break

                htD = jp.ht_dc[s.td]
                htA = jp.ht_ac[s.ta]
                coeff8, new_prev = decode_one_block(br, htD, htA, prev_dc[s.cid], zigzag_on)
                prev_dc[s.cid] = new_prev
                blocks[s.cid][by, bx] = coeff8
            else:
                # finished this MCU normally
                continue

            # If we broke because of RST mid-MCU, redo MCU at same (by,bx)
            # The simplest: decrement bx and let loop increment back.
            bx -= 1

    meta = {
        "width": W,
        "height": H,
        "qtables": jp.qtables,
        "components": jp.components,
        "sos_specs": jp.sos_specs,
        "restart_interval": jp.restart_interval,
    }
    cid_Y, cid_Cb, cid_Cr = [s.cid for s in comps]
    return blocks[cid_Y], blocks[cid_Cb], blocks[cid_Cr], meta
