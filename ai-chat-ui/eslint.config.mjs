import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": "off", // Also disable TypeScript version
      "@typescript-eslint/no-explicit-any": "off",
      "react/no-unescaped-entities": "off", // Disable rule for unescaped entities in JSX
    },
  },
];

export default eslintConfig;
