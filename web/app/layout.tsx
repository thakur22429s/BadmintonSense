import type { Metadata } from "next";
import { IBM_Plex_Sans, JetBrains_Mono } from "next/font/google";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const ibm = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-ibm",
});
const jbm = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-jbm",
});

export const metadata: Metadata = {
  title: "Badminton-Sense",
  description: "Stroke classification from monocular badminton video — pose-based deep learning. CS 535, Rutgers MS CS.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${ibm.variable} ${jbm.variable}`}>
      <body>
        <TooltipProvider>
          <Nav />
          {children}
          <Footer />
        </TooltipProvider>
      </body>
    </html>
  );
}
