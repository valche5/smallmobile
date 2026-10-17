import { useMemo } from 'preact/hooks';
import qrcode from 'qrcode-generator';

/** QR code en SVG, à scanner avec l'appareil photo de l'autre téléphone. */
export function QrCode({ text, label }: { text: string; label: string }) {
  const { n, d } = useMemo(() => {
    const qr = qrcode(0, 'M');
    qr.addData(text);
    qr.make();
    const n = qr.getModuleCount();
    let d = '';
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (qr.isDark(r, c)) d += `M${c} ${r}h1v1h-1z`;
    return { n, d };
  }, [text]);
  return (
    <svg class="qr" viewBox={`-3 -3 ${n + 6} ${n + 6}`} role="img" aria-label={label} shape-rendering="crispEdges">
      <rect x={-3} y={-3} width={n + 6} height={n + 6} fill="#fff" />
      <path d={d} fill="#1d1712" />
    </svg>
  );
}
