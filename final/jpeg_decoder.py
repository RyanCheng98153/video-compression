from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

# ============================================================
# ZigZag order
# ============================================================
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

# ============================================================
# Utilities
# ============================================================
def receive_extend(v: int, t: int) -> int:
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - ((1 << t) - 1)
    return v


# ============================================================
# Huffman
# ============================================================
@dataclass
class HuffmanTable:
    codes: Dict[Tuple[int, int], int]
    max_len: int

    def decode(self, br: "EntropyBitReader") -> int:
        code = 0
        for length in range(1, self.max_len + 1):
            code = (code << 1) | br.get_bit()
            if (length, code) in self.codes:
                return self.codes[(length, code)]
        raise ValueError("Huffman decode failed")


def build_huffman_table(lengths, symbols):
    code = 0
    k = 0
    table = {}
    max_len = 0
    for i, cnt in enumerate(lengths):
        length = i + 1
        if cnt:
            max_len = max(max_len, length)
        for _ in range(cnt):
            table[(length, code)] = symbols[k]
            k += 1
            code += 1
        code <<= 1
    return HuffmanTable(table, max_len if max_len else 16)


# ============================================================
# Exceptions
# ============================================================
class ScanTerminated(Exception):
    """Entropy-coded segment terminated by marker (EOI / next marker)."""
    pass


# ============================================================
# Entropy Bit Reader
# ============================================================
class EntropyBitReader:
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

    def _read_byte(self):
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
                    # stuffed FF
                    b = 0xFF
                elif 0xD0 <= nxt <= 0xD7:
                    # restart marker
                    self.pending_rst = nxt
                    self.reset_bits()
                    return
                else:
                    # scan terminated by marker (EOI / next segment)
                    raise ScanTerminated()

            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_cnt += 8

    def get_bits(self, n):
        if n == 0:
            return 0
        if self.pending_rst is not None:
            raise RuntimeError("RST pending")

        while self.bit_cnt < n:
            self._fill()
            if self.pending_rst is not None:
                raise RuntimeError("RST pending")

        if self.bit_cnt < n:
            raise EOFError()

        shift = self.bit_cnt - n
        val = (self.bit_buf >> shift) & ((1 << n) - 1)
        self.bit_cnt -= n
        self.bit_buf &= (1 << self.bit_cnt) - 1 if self.bit_cnt > 0 else 0
        return val

    def get_bit(self):
        return self.get_bits(1)


# ============================================================
# JPEG Parser (DQT / DHT / SOF / SOS)
# ============================================================
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


class JPEGParser:
    def __init__(self, jpeg_bytes: bytes):
        self.b = jpeg_bytes
        self.qtables = {}
        self.ht_dc = {}
        self.ht_ac = {}
        self.components = {}
        self.sos_specs = []
        self.width = 0
        self.height = 0
        self.scan_data = b""

    def parse(self):
        b = self.b
        if b[:2] != b"\xFF\xD8":
            raise ValueError("Not JPEG")

        i = 2
        while i < len(b):
            if b[i] != 0xFF:
                i += 1
                continue
            while b[i] == 0xFF:
                i += 1
            marker = b[i]; i += 1

            if marker == 0xD9:  # EOI
                break

            if marker == 0xDA:  # SOS
                length = (b[i] << 8) | b[i+1]
                seg = b[i+2:i+length]
                i += length
                self._parse_sos(seg)
                self.scan_data = self._extract_scan_data(b, i)
                break

            length = (b[i] << 8) | b[i+1]
            seg = b[i+2:i+length]
            i += length

            if marker == 0xDB:
                self._parse_dqt(seg)
            elif marker == 0xC4:
                self._parse_dht(seg)
            elif marker == 0xC0:
                self._parse_sof0(seg)

    def _parse_dqt(self, seg):
        j = 0
        while j < len(seg):
            tq = seg[j] & 0x0F
            j += 1
            qt = seg[j:j+64]
            j += 64
            q8 = np.zeros((8,8), np.float32)
            for k,(r,c) in enumerate(ZIGZAG_POS):
                q8[r,c] = qt[k]
            self.qtables[tq] = q8

    def _parse_dht(self, seg):
        j = 0
        while j < len(seg):
            tc = (seg[j] >> 4) & 0x0F
            th = seg[j] & 0x0F
            j += 1
            lengths = list(seg[j:j+16])
            j += 16
            total = sum(lengths)
            symbols = list(seg[j:j+total])
            j += total
            ht = build_huffman_table(lengths, symbols)
            if tc == 0:
                self.ht_dc[th] = ht
            else:
                self.ht_ac[th] = ht

    def _parse_sof0(self, seg):
        self.height = (seg[1] << 8) | seg[2]
        self.width  = (seg[3] << 8) | seg[4]
        nf = seg[5]
        j = 6
        for _ in range(nf):
            cid = seg[j]
            hv = seg[j+1]
            tq = seg[j+2]
            j += 3
            self.components[cid] = Component(cid, hv>>4, hv&0x0F, tq)

    def _parse_sos(self, seg):
        ns = seg[0]
        j = 1
        for _ in range(ns):
            cid = seg[j]
            tdta = seg[j+1]
            j += 2
            self.sos_specs.append(
                SOSComponentSpec(cid, tdta>>4, tdta&0x0F)
            )

    def _extract_scan_data(self, b, i):
        out = bytearray()
        while i < len(b):
            if b[i] != 0xFF:
                out.append(b[i])
                i += 1
                continue
            if i+1 >= len(b):
                break
            if b[i+1] == 0x00 or (0xD0 <= b[i+1] <= 0xD7):
                out.extend(b[i:i+2])
                i += 2
                continue
            break
        return bytes(out)


# ============================================================
# Block decode
# ============================================================
def decode_one_block(br, ht_dc, ht_ac, prev_dc, zigzag_on):
    coeff = np.zeros((8,8), np.int16)

    s = ht_dc.decode(br)
    v = br.get_bits(s)
    dc = prev_dc + receive_extend(v, s)
    prev_dc = dc
    coeff[0,0] = dc

    k = 1
    while k < 64:
        rs = ht_ac.decode(br)
        if rs == 0:
            break
        if rs == 0xF0:
            k += 16
            continue
        run = rs >> 4
        size = rs & 0x0F
        k += run
        if k >= 64:
            break
        v = br.get_bits(size)
        ac = receive_extend(v, size)
        if zigzag_on:
            r,c = ZIGZAG_POS[k]
        else:
            r,c = divmod(k, 8)
        coeff[r,c] = ac
        k += 1

    return coeff.astype(np.float32), prev_dc


# ============================================================
# Main decode
# ============================================================
def decode_baseline_huffman(jpeg_bytes: bytes, zigzag_on=True):
    jp = JPEGParser(jpeg_bytes)
    jp.parse()

    comps = jp.sos_specs
    H, W = jp.height, jp.width
    bh, bw = (H+7)//8, (W+7)//8

    blocks = {s.cid: np.zeros((bh,bw,8,8), np.float32) for s in comps}
    prev_dc = {s.cid: 0 for s in comps}
    br = EntropyBitReader(jp.scan_data)

    for by in range(bh):
        for bx in range(bw):
            for s in comps:
                while True:
                    try:
                        coeff, p = decode_one_block(
                            br, jp.ht_dc[s.td], jp.ht_ac[s.ta],
                            prev_dc[s.cid], zigzag_on
                        )
                        prev_dc[s.cid] = p
                        blocks[s.cid][by,bx] = coeff
                        break
                    except RuntimeError:
                        # RST
                        for k in prev_dc:
                            prev_dc[k] = 0
                        br.pending_rst = None
                        br.reset_bits()
                    except ScanTerminated:
                        cid = [x.cid for x in comps]
                        return blocks[cid[0]], blocks[cid[1]], blocks[cid[2]], {}

    cid = [x.cid for x in comps]
    return blocks[cid[0]], blocks[cid[1]], blocks[cid[2]], {}
