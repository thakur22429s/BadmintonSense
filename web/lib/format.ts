export const pct = (x: number, digits = 1) =>
  `${(x * 100).toFixed(digits)}%`;

export const fmt = (x: number, digits = 3) => x.toFixed(digits);

export const compactInt = (x: number) =>
  x >= 1000 ? `${(x / 1000).toFixed(1)}k` : `${x}`;
