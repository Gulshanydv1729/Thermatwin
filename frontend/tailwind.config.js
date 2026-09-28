/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Near-black slate industrial base
        scada: {
          950: "#05080c",
          900: "#0a0f16",
          850: "#0e141d",
          800: "#131b26",
          700: "#1c2634",
          600: "#283546",
          500: "#3a4a5f",
        },
        // Semantic operational colours
        amber: {
          scada: "#f5a524",
        },
        signal: {
          green: "#22c55e",
          amber: "#f5a524",
          red: "#ef4444",
          blue: "#3b9dff",
          cyan: "#22d3ee",
          violet: "#a78bfa",
        },
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
    },
  },
  plugins: [],
};
