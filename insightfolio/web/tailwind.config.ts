import type { Config } from 'tailwindcss'
export default {
content: [
'./app/**/*.{ts,tsx}',
'./components/**/*.{ts,tsx}',
],
theme: {
extend: {
colors: {
ink: '#0f172a',
subtle: '#94a3b8',
card: '#0b1220',
accent: '#22d3ee'
},
boxShadow: {
soft: '0 10px 30px rgba(0,0,0,0.2)'
}
}
},
plugins: []
} satisfies Config