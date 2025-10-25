'use client'
<Stat label="Last Close" value={fc?.last_close?.toFixed(2) ?? '—'} />
<Stat label="1d Forecast" value={kpis.one.toFixed(2)} sub={(kpis.delta1>=0?'+':'') + kpis.delta1.toFixed(2) + '% vs last'} />
<Stat label="5d Forecast" value={kpis.five.toFixed(2)} sub={(kpis.delta5>=0?'+':'') + kpis.delta5.toFixed(2) + '% vs last'} />
<Stat label="Signal" value={kpis.signal} />
</div>
)}


<div className="rounded-2xl bg-card p-4 border border-slate-800 shadow-soft">
<div className="text-sm text-subtle mb-2">Price & 5‑Day Forecast</div>
<div className="h-72">
<ResponsiveContainer width="100%" height="100%">
<LineChart data={chartData} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
<CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
<XAxis dataKey="date" tick={{ fill: '#94a3b8' }} hide={false} interval={Math.max(0, Math.floor((chartData.length-1)/8))} />
<YAxis tick={{ fill: '#94a3b8' }} domain={[ 'auto', 'auto' ]} />
<Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #1e293b' }} labelStyle={{ color: '#94a3b8' }} />
<Legend wrapperStyle={{ color: '#94a3b8' }} />
<Line type="monotone" dataKey="Close" stroke="#22d3ee" dot={false} strokeWidth={1.5} />
<Line type="monotone" dataKey="Forecast" stroke="#a78bfa" dot={{ r: 2 }} strokeDasharray="5 4" strokeWidth={1.5} />
</LineChart>
</ResponsiveContainer>
</div>
</div>


{fc && (
<div className="grid md:grid-cols-2 gap-4 mt-6">
<div className="rounded-2xl bg-card p-4 border border-slate-800">
<div className="text-sm text-subtle mb-2">Per‑Model Contributions</div>
<table className="w-full text-sm">
<thead className="text-subtle">
<tr>
<th className="text-left py-1">Horizon</th>
<th className="text-right py-1">Ensemble</th>
{Object.keys(fc.models).map(m => <th key={m} className="text-right py-1">{m.toUpperCase()}</th>)}
</tr>
</thead>
<tbody>
{fc.horizons.map((h, i) => (
<tr key={h} className="border-t border-slate-800/60">
<td className="py-1">{h}d</td>
<td className="py-1 text-right">{fc.ensemble[i].toFixed(2)}</td>
{Object.keys(fc.models).map(m => (
<td key={m} className="py-1 text-right">{fc.models[m][i]?.toFixed(2) ?? '—'}</td>
))}
</tr>
))}
</tbody>
</table>
</div>


<div className="rounded-2xl bg-card p-4 border border-slate-800">
<div className="text-sm text-subtle mb-2">Model Weights</div>
<div className="flex gap-3">
{Object.entries(fc.models).map(([m]) => (
<div key={m} className="flex-1">
<div className="text-xs text-subtle mb-1">{m.toUpperCase()}</div>
<div className="h-2 bg-slate-800 rounded-full">
{/* simple visual weight hint via first-horizon absolute */}
<div className={clsx("h-2 rounded-full bg-accent")}
style={{ width: `${Math.min(100, Math.abs((fc.models[m][0] ?? 0) / (fc.ensemble[0] || 1)) * 100)}%` }} />
</div>
</div>
))}
</div>
</div>
</div>
)}
</main>
</div>
)
}