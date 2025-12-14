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
    return (be_bytes[i] << 8) | be_bytes[i + 1], i + 2


# ---------------------------
# Huffman
# ---------------------------
@dataclass
class HuffmanTable:
    codes: Dict[Tuple[int, int], int]   # (length, code)->symbol
    max_len: int

    def decode(self, br: "EntropyBitReader") -> int:
        code = 0
        for length in range(1, self.max_len + 1):
            code = (code << 1) | br.get_bit()
            sym = self.codes.get((length, code), None)
            if sym is not None:
                return sym
        raise ValueError("Huffman decode failed")


def build_huffman_table(lengths: List[int], symbols: List[int]) -> HuffmanTable:
    """
    Canonical Huffman code building per JPEG spec.
    lengths: 16 counts for code length 1..16
    symbols: symbols in order
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
# Entropy reader w/ RST
# ---------------------------
class EntropyBitReader:
    """
    Read bits from entropy-coded segment (the scan data only), with:
      - byte stuffing: 0xFF 0x00 -> data byte 0xFF
      - restart markers: 0xFF D0..D7
    IMPORTANT: scan_data passed in here must NOT include markers like SOS header, DQT, etc.
    """
    def __init__(self, scan_data: bytes):
        self.data = scan_data
        self.pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0
        self.pending_rst: Optional[int] = None  # 0xD0..0xD7
        self.done = False

    def reset_bits(self):
        self.bit_buf = 0
        self.bit_cnt = 0

    def _read_byte(self) -> Optional[int]:
        if self.pos >= len(self.data):
            return None
        b = self.data[self.pos]
        self.pos += 1
        return b

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
                    # stuffed 0xFF
                    b = 0xFF
                elif 0xD0 <= nxt <= 0xD7:
                    # RST marker inside scan
                    self.pending_rst = nxt
                    self.reset_bits()
                    return
                else:
                    # In pure scan data, the only legal 0xFF is stuffed or RST.
                    # If we see something else, stop; caller will fail clearly.
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
# JPEG parser w/ DRI + correct scan extraction
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
        self.restart_interval: int = 0  # from DRI (in MCUs)
        self.scan_data: bytes = b""     # entropy-coded segment ONLY (no EOI)

    def parse(self):
        b = self.b
        i = 0
        if b[0:2] != b"\xFF\xD8":
            raise ValueError("Not a JPEG (missing SOI)")
        i = 2

        def read_marker():
            nonlocal i
            # markers start with 0xFF, may have fill 0xFFs
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

            if m in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7):
                # standalone RST outside scan (rare)
                continue

            if m == 0xDA:  # SOS
                seglen, i = _read_u16(b, i)
                seg = b[i:i + seglen - 2]
                i += seglen - 2
                self._parse_sos(seg)

                # After SOS header, scan data continues until the next marker (EOI usually),
                # but note: inside scan, 0xFF bytes are escaped as 0xFF 0x00, and RST markers are allowed.
                self.scan_data, i = self._extract_scan_data(b, i)
                break

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
                pass

        if not self.scan_data:
            raise ValueError("Missing SOS/scan data")

    def _parse_dri(self, seg: bytes):
        # length=4, then Ri (2 bytes)
        ri = (seg[0] << 8) | seg[1]
        self.restart_interval = int(ri)

    def _parse_dqt(self, seg: bytes):
        j = 0
        while j < len(seg):
            pq_tq = seg[j]; j += 1
            pq = (pq_tq >> 4) & 0x0F
            tq = pq_tq & 0x0F
            if pq != 0:
                raise NotImplementedError("16-bit quant table not supported")
            qt = np.frombuffer(seg[j:j + 64], dtype=np.uint8).astype(np.float32)
            j += 64

            q8 = np.zeros((8, 8), dtype=np.float32)
            for k, (r, c) in enumerate(ZIGZAG_POS):
                q8[r, c] = qt[k]
            self.qtables[tq] = q8

    def _parse_dht(self, seg: bytes):
        j = 0
        while j < len(seg):
            tc_th = seg[j]; j += 1
            tc = (tc_th >> 4) & 0x0F  # 0=DC,1=AC
            th = tc_th & 0x0F
            lengths = list(seg[j:j + 16]); j += 16
            total = sum(lengths)
            symbols = list(seg[j:j + total]); j += total
            ht = build_huffman_table(lengths, symbols)
            if tc == 0:
                self.ht_dc[th] = ht
            else:
                self.ht_ac[th] = ht

    def _parse_sof0(self, seg: bytes):
        p = seg[0]
        if p != 8:
            raise NotImplementedError("Only 8-bit precision supported")
        self.height = (seg[1] << 8) | seg[2]
        self.width  = (seg[3] << 8) | seg[4]
        nf = seg[5]
        j = 6
        self.components = {}
        for _ in range(nf):
            cid = seg[j]
            hv = seg[j + 1]
            tq = seg[j + 2]
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
            tdta = seg[j + 1]
            j += 2
            td = (tdta >> 4) & 0x0F
            ta = tdta & 0x0F
            specs.append(SOSComponentSpec(cid=cid, td=td, ta=ta))
        self.sos_specs = specs
        # ignore Ss/Se/AhAl for baseline

    def _extract_scan_data(self, b: bytes, start: int) -> Tuple[bytes, int]:
        """
        Extract entropy-coded scan data bytes from JPEG stream starting at 'start'
        until next marker (EOI or next segment).
        Must respect:
          - 0xFF00 is stuffed data byte 0xFF (NOT marker)
          - 0xFFD0..FFD7 are restart markers inside scan (part of scan stream)
        We return raw scan bytes including stuffed sequences and RST markers,
        but NOT including the final marker (like EOI).
        """
        out = bytearray()
        i = start
        while i < len(b):
            x = b[i]
            if x != 0xFF:
                out.append(x)
                i += 1
                continue

            # x==0xFF
            if i + 1 >= len(b):
                break
            y = b[i + 1]

            if y == 0x00:
                # stuffed 0xFF
                out.append(0xFF)
                i += 2
                continue

            if 0xD0 <= y <= 0xD7:
                # restart marker in scan
                out.append(0xFF)
                out.append(y)
                i += 2
                continue

            # Otherwise, it's a marker that ends scan (EOI or another segment)
            break

        return bytes(out), i


# ---------------------------
# Block decode
# ---------------------------
def decode_one_block(
    br: EntropyBitReader,
    ht_dc: HuffmanTable,
    ht_ac: HuffmanTable,
    prev_dc: int,
    zigzag_on: bool
) -> Tuple[np.ndarray, int]:
    coeff = np.zeros((8, 8), dtype=np.int16)

    # DC
    s = ht_dc.decode(br)
    v = br.get_bits(s)
    diff = receive_extend(v, s)
    dc = prev_dc + diff
    prev_dc = dc

    if zigzag_on:
        r, c = ZIGZAG_POS[0]
        coeff[r, c] = dc
    else:
        coeff[0, 0] = dc

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
            v = br.get_bits(size)
            ac = receive_extend(v, size)
        else:
            ac = 0

        if zigzag_on:
            r, c = ZIGZAG_POS[k]
            coeff[r, c] = ac
        else:
            rr = k // 8
            cc = k % 8
            coeff[rr, cc] = ac

        k += 1

    return coeff.astype(np.float32), prev_dc


def decode_baseline_huffman(jpeg_bytes: bytes, zigzag_on: bool = True):
    """
    Baseline JPEG entropy decode (Huffman) with real scan extraction and RST support.
    Assumptions for course project:
      - 3 components in one scan (YCbCr)
      - 4:4:4 (h=v=1 for all components)  [same as your current constraint]
    """
    jp = JPEGParser(jpeg_bytes)
    jp.parse()

    comps = jp.sos_specs
    if len(comps) != 3:
        raise NotImplementedError("Expect 3 components in one scan (YCbCr)")

    # enforce 4:4:4
    for s in comps:
        c = jp.components[s.cid]
        if not (c.h == 1 and c.v == 1):
            raise NotImplementedError("Subsampling not supported; encode with subsampling=0 (4:4:4).")

    H, W = jp.height, jp.width
    bw = (W + 7) // 8
    bh = (H + 7) // 8

    blocks = {s.cid: np.zeros((bh, bw, 8, 8), dtype=np.float32) for s in comps}
    prev_dc = {s.cid: 0 for s in comps}

    br = EntropyBitReader(jp.scan_data)

    # Restart handling:
    # - If DRI exists, restart interval is jp.restart_interval (MCUs)
    # - Even if DRI not present, we still handle actual RST markers when encountered.
    ri = jp.restart_interval
    mcu_count = 0

    def reset_predictors_and_align():
        for cid in prev_dc:
            prev_dc[cid] = 0
        br.pending_rst = None
        br.reset_bits()

    for by in range(bh):
        for bx in range(bw):

            # If a restart interval is declared, we expect an RST marker every 'ri' MCUs.
            # Decoder behavior: if marker appears, reset; if absent, continue (some encoders omit DRI).
            if ri > 0 and mcu_count > 0 and (mcu_count % ri == 0):
                # We are at a restart boundary; consume marker if present by forcing _fill
                # until pending_rst is set or we get data. If pending_rst set, reset state.
                try:
                    br._fill()
                except Exception:
                    pass
                if br.pending_rst is not None:
                    reset_predictors_and_align()

            # Also handle markers encountered naturally
            if br.pending_rst is not None:
                reset_predictors_and_align()

            # Decode MCU (Y, Cb, Cr) in scan order
            try:
                for s in comps:
                    htD = jp.ht_dc[s.td]
                    htA = jp.ht_ac[s.ta]
                    coeff8, prev = decode_one_block(br, htD, htA, prev_dc[s.cid], zigzag_on)
                    prev_dc[s.cid] = prev
                    blocks[s.cid][by, bx] = coeff8
            except (RuntimeError, EOFError, ValueError) as e:
                # If we desync due to a marker boundary, try one recovery:
                # force-fill to detect RST and reset, then retry this MCU once.
                br._fill()
                if br.pending_rst is not None:
                    reset_predictors_and_align()
                    for s in comps:
                        htD = jp.ht_dc[s.td]
                        htA = jp.ht_ac[s.ta]
                        coeff8, prev = decode_one_block(br, htD, htA, prev_dc[s.cid], zigzag_on)
                        prev_dc[s.cid] = prev
                        blocks[s.cid][by, bx] = coeff8
                else:
                    raise

            mcu_count += 1

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
