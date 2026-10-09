/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#eef2f8',
          100: '#d6deec',
          200: '#adbdd9',
          300: '#7f97c0',
          400: '#5272a3',
          500: '#365585',
          600: '#27416a',
          700: '#1c3153',
          800: '#13233d',
          900: '#0b1729',
          950: '#060d19',
        },
        teal: {
          50: '#effaf8',
          100: '#d4f1ec',
          200: '#a9e3da',
          300: '#73cdc1',
          400: '#44b1a4',
          500: '#2a958a',
          600: '#1f786f',
          700: '#1c605a',
          800: '#1a4d49',
          900: '#19403d',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(11,23,41,0.06), 0 4px 12px rgba(11,23,41,0.06)',
      },
    },
  },
  plugins: [],
};
