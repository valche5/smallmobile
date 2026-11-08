"""
Police « Pièce SmallMobile » : un seul caractère, 💰 (U+1F4B0), dessiné comme la pièce de score du jeu
(pièce d'or, cercle intérieur, un « 1 » au centre). Placée en tête de --font et --emoji (styles.css), elle
remplace l'emoji bourse partout dans le jeu, sans toucher aux textes.

Couleurs : table COLR v0 + CPAL (lue par Chrome, Safari et Firefox).
Produit src/css/coin-font.css (police intégrée en base64, importée par styles.css).

    pip install fonttools
    python src/assets/fonts/coin-font.py
"""

import base64
import io
import math
from pathlib import Path

from fontTools.colorLib.builder import buildCOLR, buildCPAL
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPM = 1000
CX, CY = 500, 330  # centre de la pièce (sur la ligne de base des emojis, un peu au-dessus)


def circle(pen, r, clockwise=True, n=72):
    """Un cercle en polygone (n côtés) ; sens horaire = plein, antihoraire = trou (règle non nulle)."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((round(CX + r * math.cos(a)), round(CY + r * math.sin(a))))
    if clockwise:
        pts.reverse()
    pen.moveTo(pts[0])
    for p in pts[1:]:
        pen.lineTo(p)
    pen.closePath()


def poly(pen, pts):
    """Un polygone donné dans le sens horaire."""
    pen.moveTo(pts[0])
    for p in pts[1:]:
        pen.lineTo(p)
    pen.closePath()


def glyph(draw):
    pen = TTGlyphPen(None)
    draw(pen)
    return pen.glyph()


ONE_SCALE = 0.68  # taille du « 1 » sur la pièce (1 = toute la hauteur intérieure)


def one(pen):
    """Le « 1 » : la barre, le petit drapeau en haut à gauche et le socle, réduit autour du centre de la pièce."""
    def s(pts):
        # le chiffre dessiné est centré vers (505, 348) : on le ramène au centre de la pièce en le réduisant
        return [(round(CX + (x - 505) * ONE_SCALE), round(CY + (y - 348) * ONE_SCALE)) for x, y in pts]

    poly(pen, s([(455, 625), (575, 625), (575, 135), (455, 135)]))  # barre
    poly(pen, s([(455, 625), (480, 555), (360, 500), (335, 565)]))  # drapeau
    poly(pen, s([(380, 175), (650, 175), (650, 70), (380, 70)]))  # socle


glyphs = {
    '.notdef': glyph(lambda p: None),
    'coin': glyph(lambda p: circle(p, 450)),  # repli sans couleurs : un disque
    'coin.edge': glyph(lambda p: circle(p, 450)),
    'coin.face': glyph(lambda p: circle(p, 402)),
    'coin.ring': glyph(lambda p: (circle(p, 345), circle(p, 318, clockwise=False))),
    'coin.one': glyph(one),
}
order = list(glyphs)

fb = FontBuilder(UPM, isTTF=True)
fb.setupGlyphOrder(order)
fb.setupCharacterMap({0x1F4B0: 'coin'})
fb.setupGlyf(glyphs)
# l'approche gauche de chaque glyphe = son xMin : sinon chaque couche serait recalée sur son propre bord gauche
glyf = fb.font['glyf']
fb.setupHorizontalMetrics({g: (1000 if g != '.notdef' else 500, getattr(glyf[g], 'xMin', 0)) for g in order})
fb.setupHorizontalHeader(ascent=900, descent=-200)
fb.setupNameTable({'familyName': 'Piece SmallMobile', 'styleName': 'Regular'})
fb.setupOS2(sTypoAscender=900, sTypoDescender=-200, usWinAscent=900, usWinDescent=200)
fb.setupPost()
fb.font['CPAL'] = buildCPAL([[
    (0x8A / 255, 0x5A / 255, 0x00 / 255, 1.0),  # 0 bord
    (0xF7 / 255, 0xC8 / 255, 0x43 / 255, 1.0),  # 1 or
    (0xC9 / 255, 0x8A / 255, 0x12 / 255, 1.0),  # 2 cercle intérieur
    (0x6B / 255, 0x43 / 255, 0x00 / 255, 1.0),  # 3 chiffre
]])
fb.font['COLR'] = buildCOLR({'coin': [('coin.edge', 0), ('coin.face', 1), ('coin.ring', 2), ('coin.one', 3)]})

buf = io.BytesIO()
fb.font.save(buf)
data = base64.b64encode(buf.getvalue()).decode()
out = Path(__file__).resolve().parents[2] / 'css' / 'coin-font.css'
out.write_text(
    '/* Généré par src/assets/fonts/coin-font.py : 💰 dessiné comme la pièce de score (avec un « 1 »). */\n'
    '@font-face {\n'
    "  font-family: 'Piece SmallMobile';\n"
    f"  src: url(data:font/ttf;base64,{data}) format('truetype');\n"
    '  unicode-range: U+1F4B0;\n'
    '  font-display: block;\n'
    '}\n',
    encoding='utf-8',
)
print(f'{out} : {len(buf.getvalue())} octets')
