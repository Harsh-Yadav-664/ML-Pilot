import { ButtonHTMLAttributes, ReactNode, useEffect, useId, useRef, useState } from 'react';
import { cn } from './utils/cn';
import { ExperimentStatus } from './types';

/* ------------------------------------------------------------- hooks */
export function useCountUp(target: number, duration = 900) {
  const [v, setV] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const step = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - p, 3);
      const val = a + (target - a) * e;
      from.current = val;
      setV(val);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return v;
}

export function useNow(ms = 15000) {
  const [, set] = useState(0);
  useEffect(() => {
    const t = setInterval(() => set((n) => n + 1), ms);
    return () => clearInterval(t);
  }, [ms]);
}

export function Num({
  value,
  decimals = 0,
  prefix = '',
  suffix = '',
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
}) {
  const v = useCountUp(value);
  return (
    <>
      {prefix}
      {v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      {suffix}
    </>
  );
}

/* ------------------------------------------------------------- status */
export const STATUS: Record<ExperimentStatus, { label: string; hex: string }> = {
  completed: { label: 'Completed', hex: '#b2cd8b' },
  running: { label: 'Running', hex: '#f7cd82' },
  failed: { label: 'Failed', hex: '#e0705a' },
  queued: { label: 'Queued', hex: '#a08d76' },
};

/** Square LED. The whole UI uses squares where other tools use dots. */
export function Led({ color = '#a08d76', pulse, className }: { color?: string; pulse?: boolean; className?: string }) {
  return (
    <span className={cn('relative inline-block h-1.5 w-1.5 shrink-0', className)} style={{ background: color }}>
      {pulse && <span className="absolute inset-0 animate-ping" style={{ background: color, opacity: 0.55 }} />}
    </span>
  );
}

export function StatusChip({ status }: { status: ExperimentStatus }) {
  const s = STATUS[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 font-mono text-[10px] font-medium uppercase tracking-[0.1em]"
      style={{ color: s.hex }}
    >
      <Led color={s.hex} pulse={status === 'running'} />
      {s.label}
    </span>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn('font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-mute', className)}>
      {children}
    </span>
  );
}

export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        'inline-flex h-[18px] min-w-[18px] items-center justify-center border border-rule px-1 font-mono text-[10px] text-mute',
        className
      )}
    >
      {children}
    </kbd>
  );
}

type TagTone = 'neutral' | 'copper' | 'sage' | 'clay';
export function Tag({ children, tone = 'neutral', className }: { children: ReactNode; tone?: TagTone; className?: string }) {
  const tones: Record<TagTone, string> = {
    neutral: 'border-rule text-mute',
    copper: 'border-copper/60 text-copper-2',
    sage: 'border-sage/50 text-sage',
    clay: 'border-clay/60 text-clay',
  };
  return (
    <span
      className={cn(
        'inline-flex h-[18px] items-center border px-1.5 font-mono text-[9.5px] font-medium uppercase tracking-[0.12em]',
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

export function Delta({ value, digits = 3, className }: { value: number; digits?: number; className?: string }) {
  const up = value > 0.0004;
  const down = value < -0.0004;
  return (
    <span
      className={cn(
        'font-mono text-[11px] font-medium tabular-nums',
        up && 'text-sage',
        down && 'text-clay',
        !up && !down && 'text-mute',
        className
      )}
    >
      {up ? '▲ +' : down ? '▼ −' : '■ '}
      {Math.abs(value).toFixed(digits)}
    </span>
  );
}

/* ------------------------------------------------------------- buttons */
type Variant = 'primary' | 'outline' | 'ghost' | 'danger';
export function Btn({
  variant = 'outline',
  size = 'md',
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg' }) {
  const v: Record<Variant, string> = {
    primary: 'chamfer bg-copper text-ink hover:bg-copper-2',
    outline:
      'border border-rule text-bone hover:-translate-x-px hover:-translate-y-px hover:border-copper hover:text-copper-2 hover:shadow-[3px_3px_0_0_rgba(224,162,79,0.28)]',
    ghost: 'text-mute hover:bg-panel-2 hover:text-bone',
    danger: 'border border-clay/50 text-clay hover:bg-clay/10',
  };
  const s = { sm: 'h-7 px-2.5 text-[10.5px]', md: 'h-9 px-4 text-[11px]', lg: 'h-11 px-5 text-[11.5px]' };
  return (
    <button
      {...rest}
      className={cn(
        'inline-flex shrink-0 select-none items-center justify-center gap-2 whitespace-nowrap font-mono font-semibold uppercase tracking-[0.1em] transition-[transform,background-color,border-color,color,box-shadow] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-40',
        v[variant],
        s[size],
        className
      )}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------- layout */
export function Panel({ className, children, ticks }: { className?: string; children: ReactNode; ticks?: boolean }) {
  return <div className={cn('relative border border-line bg-panel/80', ticks && 'ticks', className)}>{children}</div>;
}
export const Card = Panel;

export function CardHeader({ title, right, sub }: { title: string; sub?: string; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
      <div className="flex min-w-0 items-baseline gap-2.5">
        <h3 className="font-display text-[17px] font-bold uppercase leading-none tracking-wide text-bone">{title}</h3>
        {sub && <span className="truncate font-mono text-[10.5px] text-mute">{sub}</span>}
      </div>
      {right}
    </div>
  );
}

/** Numbered editorial section header with a rule that runs to the edge. */
export function SectionHead({ n, title, right }: { n: string; title: string; right?: ReactNode }) {
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[10.5px] font-medium text-copper">{n}</span>
      <h2 className="font-display text-[24px] font-bold uppercase leading-none tracking-wide text-bone">{title}</h2>
      <span className="h-px flex-1 bg-line" />
      {right}
    </div>
  );
}

/** Label ........ value, the dotted leader of a printed spec sheet. */
export function Leader({ k, v, className }: { k: string; v: ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-baseline gap-2 py-[5px]', className)}>
      <span className="text-[12px] text-mute">{k}</span>
      <span className="min-w-3 flex-1 -translate-y-[3px] border-b border-dotted border-rule" />
      <span className="font-mono text-[12px] text-bone">{v}</span>
    </div>
  );
}

/** Instrument-style segmented gauge. */
export function Seg({
  value,
  total = 30,
  color = '#e0a24f',
  height = 14,
  className,
}: {
  value: number;
  total?: number;
  color?: string;
  height?: number;
  className?: string;
}) {
  const on = Math.round((Math.max(0, Math.min(100, value)) / 100) * total);
  return (
    <div className={cn('flex gap-[2px]', className)} aria-hidden>
      {Array.from({ length: total }).map((_, i) => (
        <span
          key={i}
          className="seg-in flex-1 transition-colors duration-500"
          style={{
            height,
            background: i < on ? color : '#221a13',
            animationDelay: `${i * 14}ms`,
            transitionDelay: `${i * 10}ms`,
          }}
        />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------- charts */
export interface Point {
  id: string;
  y: number;
  label: string;
}

export function LineChart({
  points,
  baseline,
  onSelect,
  color = '#e0a24f',
  height = 180,
  yLabel,
}: {
  points: Point[];
  baseline?: number;
  onSelect?: (id: string) => void;
  color?: string;
  height?: number;
  yLabel?: boolean;
}) {
  const W = 560,
    H = height,
    pl = yLabel === false ? 10 : 36,
    pr = 12,
    pt = 14,
    pb = 22;
  const [hover, setHover] = useState<number | null>(null);
  const pid = `hatch-${useId().replace(/:/g, '')}`;
  if (!points.length) return <div className="flex h-44 items-center justify-center text-[12px] text-mute">No points yet</div>;

  const ys = points.map((p) => p.y).concat(baseline !== undefined ? [baseline] : []);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const pad = (hi - lo) * 0.14 || 0.02;
  const y0 = lo - pad;
  const y1 = hi + pad;
  const X = (i: number) => pl + (points.length === 1 ? (W - pl - pr) / 2 : (i / (points.length - 1)) * (W - pl - pr));
  const Y = (v: number) => pt + (1 - (v - y0) / (y1 - y0)) * (H - pt - pb);

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${X(i)},${Y(p.y)}`).join(' ');
  const area = `${line} L${X(points.length - 1)},${H - pb} L${X(0)},${H - pb} Z`;
  const ticks = [0, 1, 2, 3].map((i) => y0 + ((y1 - y0) * i) / 3);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full overflow-visible">
      <defs>
        {/* engraved hatching instead of a gradient wash */}
        <pattern id={pid} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke={color} strokeWidth="1" opacity="0.4" />
        </pattern>
      </defs>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pl} x2={W - pr} y1={Y(t)} y2={Y(t)} stroke="#2a2018" />
          {yLabel !== false && (
            <text x={pl - 8} y={Y(t) + 3} textAnchor="end" className="fill-mute font-mono" fontSize="9">
              {t.toFixed(2)}
            </text>
          )}
        </g>
      ))}
      {baseline !== undefined && (
        <g>
          <line x1={pl} x2={W - pr} y1={Y(baseline)} y2={Y(baseline)} stroke="#a08d76" strokeDasharray="2 4" />
          <text x={W - pr} y={Y(baseline) - 5} textAnchor="end" className="fill-mute font-mono" fontSize="8.5">
            {yLabel === false ? 'LIMIT' : 'BASELINE'}
          </text>
        </g>
      )}
      <path d={area} fill={`url(#${pid})`} />
      <path key={`s${points.length}${color}`} d={line} pathLength={1} fill="none" stroke={color} strokeWidth="1.75" strokeLinejoin="miter" className="draw" />
      {points.map((p, i) => {
        const s = hover === i ? 10 : 6;
        return (
          <g key={p.id}>
            <rect x={X(i) - s / 2} y={Y(p.y) - s / 2} width={s} height={s} fill="#0c0907" stroke={color} strokeWidth="1.5" style={{ transition: 'all .12s' }} />
            <rect
              x={X(i) - 12}
              y={Y(p.y) - 12}
              width={24}
              height={24}
              fill="transparent"
              className="cursor-pointer"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect?.(p.id)}
            />
          </g>
        );
      })}
      {hover !== null && (
        <g className="pointer-events-none">
          <rect x={Math.min(Math.max(X(hover) - 52, 0), W - 104)} y={Y(points[hover].y) - 36} width="104" height="22" fill="#140f0b" stroke="#3f3224" />
          <text x={Math.min(Math.max(X(hover), 52), W - 52)} y={Y(points[hover].y) - 21} textAnchor="middle" className="fill-bone font-mono" fontSize="10">
            {points[hover].label} · {points[hover].y.toFixed(3)}
          </text>
        </g>
      )}
    </svg>
  );
}

export function Curve({ data, color = '#e0a24f', height = 72 }: { data: number[]; color?: string; height?: number }) {
  const W = 300;
  const lo = Math.min(...data);
  const hi = Math.max(...data);
  const pid = `ch-${useId().replace(/:/g, '')}`;
  const X = (i: number) => (i / (data.length - 1)) * W;
  const Y = (v: number) => 6 + (1 - (v - lo) / (hi - lo || 1)) * (height - 12);
  const d = data.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className="w-full" style={{ height }}>
      <defs>
        <pattern id={pid} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke={color} strokeWidth="1" opacity="0.35" />
        </pattern>
      </defs>
      <path d={`${d} L${W},${height} L0,${height} Z`} fill={`url(#${pid})`} />
      <path d={d} pathLength={1} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" className="draw" />
    </svg>
  );
}

export function MiniHist({ data, muted }: { data: number[]; muted?: boolean }) {
  return (
    <div className="flex h-5 min-w-0 items-end gap-[2px]">
      {data.map((v, i) => (
        <span
          key={i}
          className={cn('min-w-0 flex-1 transition-colors', muted ? 'bg-rule' : 'bg-mute/60 group-hover:bg-copper')}
          style={{ height: `${Math.max(8, v * 100)}%`, maxWidth: 4 }}
        />
      ))}
    </div>
  );
}

export function VolumeBars({ data }: { data: { h: number; n: number }[] }) {
  const mx = Math.max(...data.map((d) => d.n), 1);
  const [h, setH] = useState<number | null>(null);
  return (
    <div className="flex h-28 items-end gap-[3px]">
      {data.map((d) => (
        <div key={d.h} className="relative flex-1" onMouseEnter={() => setH(d.h)} onMouseLeave={() => setH(null)}>
          <div
            className={cn('w-full transition-colors', h === d.h ? 'bg-copper-2' : 'bg-copper/40 hover:bg-copper')}
            style={{ height: `${(d.n / mx) * 100}%`, minHeight: 4 }}
          />
          {h === d.h && (
            <div className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap border border-rule bg-panel px-1.5 py-0.5 font-mono text-[10px] text-bone">
              {d.h}:00 · {d.n}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Scroll-triggered entrance. Falls back to visible if IO is unavailable. */
export function Reveal({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return setSeen(true);
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold: 0.05, rootMargin: '0px 0px -30px' }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={cn(
        'transition-[opacity,transform] duration-700 ease-[cubic-bezier(.22,1,.36,1)]',
        seen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0',
        className
      )}
    >
      {children}
    </div>
  );
}
