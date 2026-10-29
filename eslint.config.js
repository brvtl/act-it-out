import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'public/ort', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.strict,
);
