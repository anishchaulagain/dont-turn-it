// Word-level diff (LCS) for side-by-side paragraph comparison. Paragraphs are a few
// hundred tokens at most, so the O(n·m) table is cheap and avoids a dependency.

export type DiffPart = { text: string; kind: "same" | "removed" | "added" };

export function wordDiff(before: string, after: string): { left: DiffPart[]; right: DiffPart[] } {
  const a = before.split(/(\s+)/).filter(Boolean);
  const b = after.split(/(\s+)/).filter(Boolean);
  const n = a.length;
  const m = b.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const left: DiffPart[] = [];
  const right: DiffPart[] = [];
  const push = (list: DiffPart[], text: string, kind: DiffPart["kind"]) => {
    const last = list[list.length - 1];
    if (last && last.kind === kind) last.text += text;
    else list.push({ text, kind });
  };

  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      push(left, a[i], "same");
      push(right, b[j], "same");
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      push(left, a[i++], "removed");
    } else {
      push(right, b[j++], "added");
    }
  }
  while (i < n) push(left, a[i++], "removed");
  while (j < m) push(right, b[j++], "added");
  return { left, right };
}
