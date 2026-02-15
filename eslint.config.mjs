import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { FlatCompat } from '@eslint/eslintrc';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends('next/core-web-vitals'),
  {
    rules: {
      // Allow setState in effects (existing patterns for localStorage hydration)
      'react-hooks/set-state-in-effect': 'off',
      // Allow impure functions in state initializers (Date.now())
      'react-hooks/purity': 'off',
      // Allow <img> tags (not all images need next/image optimization)
      '@next/next/no-img-element': 'warn',
    },
  },
];

export default eslintConfig;
