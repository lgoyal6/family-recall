"use client";
import { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import Orb from "@/components/ui/components-primitives-avatar";
import {
  Siren, GitFork, ScanSearch, Grid3x3, MessagesSquare, Scale, ReceiptText, Sparkles, Map,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Incident", icon: Siren },
  { href: "/family", label: "Family", icon: GitFork },
  { href: "/screening", label: "Screening", icon: ScanSearch },
  { href: "/matrix", label: "Transfer matrix", icon: Grid3x3 },
  { href: "/evidence", label: "Evidence", icon: MessagesSquare },
  { href: "/cost", label: "Cost of the fix", icon: Scale },
  { href: "/receipts", label: "Receipts", icon: ReceiptText },
  { href: "/ask", label: "Ask the evidence", icon: Sparkles },
  { href: "/roadmap", label: "Roadmap", icon: Map },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const path = usePathname();
  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden md:flex-row">
      <Sidebar open={open} setOpen={setOpen}>
        <SidebarBody className="justify-between gap-8">
          <div className="flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
            <Link href="/" className="flex items-center gap-2.5 py-1">
              <Orb color="blue" size="sm" />
              {open && <span className="whitespace-pre text-sm font-semibold tracking-tight">Family Recall</span>}
            </Link>
            <nav className="mt-8 flex flex-col gap-1">
              {NAV.map(({ href, label, icon: Icon }) => {
                const active = href === "/" ? path === "/" : path.startsWith(href);
                return (
                  <SidebarLink
                    key={href}
                    link={{
                      label,
                      href,
                      icon: <Icon className={cn("size-5 shrink-0", active ? "text-primary" : "text-muted-foreground")} />,
                    }}
                    className={cn("rounded-md px-1.5", active && "bg-sidebar-accent")}
                  />
                );
              })}
            </nav>
          </div>
          <p className={cn("text-xs text-muted-foreground", !open && "hidden")}>
            Tiny Aya family, Lexsi Track 2
          </p>
        </SidebarBody>
      </Sidebar>
      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}
