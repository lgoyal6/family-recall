import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // standalone server bundle for the Hugging Face Space (Docker SDK, port 7860)
  output: "standalone",
  outputFileTracingIncludes: { "/**": ["./public/data/**"] },
};

export default nextConfig;
