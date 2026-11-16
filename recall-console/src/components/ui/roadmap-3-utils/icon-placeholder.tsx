"use client";
import { ChevronUp, Circle, type LucideProps } from "lucide-react";

type Props = LucideProps & {
  lucide: string;
  tabler?: string;
  hugeicons?: string;
  phosphor?: string;
  remixicon?: string;
};

const ICONS: Record<string, React.ComponentType<LucideProps>> = { ChevronUp };

/** Renders the lucide icon named by `lucide` (only the ones this app uses are bundled); other icon-set names are ignored. */
export function IconPlaceholder({ lucide, tabler: _t, hugeicons: _h, phosphor: _p, remixicon: _r, ...rest }: Props) {
  const Icon = ICONS[lucide] ?? Circle;
  return <Icon {...rest} />;
}
