/* === State Management === */
let autoRefreshEnabled = true;
let autoRefreshInterval = null;
let lastUpdateTime = null;
const REFRESH_INTERVAL = 60000; // 60 seconds

// Portfolio tracking
let activePositions = new Map(); // ticker -> {action, entry_price, entry_date, shares, note}
let currentPrices = new Map();   // ticker -> current_price

/* === tiny helpers === */
const $  = (q, el=document) => el.querySelector(q);
const $$ = (q, el=document) => [...el.querySelectorAll(q)];
const fmt = (x, d=2) => (x===null||x===undefined||Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const api = (p, opt={}) => fetch(p, opt).then(r => r.json()).catch(()=> ({}));

document.addEventListener("DOMContentLoaded", init);

async function init(){
  $("#beginBtn")?.addEventListener("click", runSearch);
  $("#chartsBtn")?.addEventListener("click", openAncientCharts);
  $("#searchBox")?.addEventListener("keydown", e => { if (e.key==="Enter") runSearch(); });

  $("#drawerClose")?.addEventListener("click", ()=> $("#detailDrawer")?.classList.remove("open"));
  $("#chartsClose")?.addEventListener("click", ()=> $("#chartsModal")?.classList.remove("open"));

  // Keyboard shortcuts
  document.addEventListener("keydown", handleKeyboardShortcuts);

  // Real-time updates
  $("#refreshBtn")?.addEventListener("click", manualRefresh);
  $("#autoRefreshToggle")?.addEventListener("change", toggleAutoRefresh);
  
  // first paint
  await runSearch();
  await refreshPlans();
  
  // Start auto-refresh
  startAutoRefresh();
  updateLastRefreshTime();
}

/* ---------- Real-Time Updates ---------- */
function startAutoRefresh() {
  if (autoRefreshInterval) {
    clearInterval(autoRefreshInterval);
  }
  
  if (autoRefreshEnabled) {
    autoRefreshInterval = setInterval(async () => {
      await refreshPrices();
      updateLastRefreshTime();
    }, REFRESH_INTERVAL);
    
    updateLiveIndicator(true);
  } else {
    updateLiveIndicator(false);
  }
}

function stopAutoRefresh() {
  if (autoRefreshInterval) {
    clearInterval(autoRefreshInterval);
    autoRefreshInterval = null;
  }
  updateLiveIndicator(false);
}

function toggleAutoRefresh(e) {
  autoRefreshEnabled = e ? e.target.checked : !autoRefreshEnabled;
  
  if (autoRefreshEnabled) {
    startAutoRefresh();
    showNotification('🔄 Auto-refresh enabled');
  } else {
    stopAutoRefresh();
    showNotification('⏸️ Auto-refresh paused');
  }
}

async function manualRefresh() {
  const btn = $("#refreshBtn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "⟳ Refreshing...";
  }
  
  await refreshPrices();
  updateLastRefreshTime();
  showNotification('✓ Prices updated');
  
  if (btn) {
    btn.disabled = false;
    btn.textContent = "⟳ Refresh";
  }
}

async function refreshPrices() {
  const searchBox = $("#searchBox");
  if (!searchBox || !searchBox.value.trim()) return;
  
  const tickers = searchBox.value.trim();
  await loadHoldings(tickers);
  
  // Also update portfolio if there are active positions
  if (activePositions.size > 0) {
    await updatePortfolioView();
  }
}

function updateLastRefreshTime() {
  lastUpdateTime = new Date();
  const el = $("#lastUpdate");
  if (el) {
    el.textContent = `Last: ${lastUpdateTime.toLocaleTimeString()}`;
  }
}

function updateLiveIndicator(isLive) {
  const indicator = $("#liveIndicator");
  if (!indicator) return;
  
  if (isLive) {
    indicator.innerHTML = '<span style="display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite;"></span>LIVE';
    indicator.style.color = '#10b981';
  } else {
    indicator.innerHTML = '<span style="display: inline-block; width: 8px; height: 8px; background: #64748b; border-radius: 50%; margin-right: 6px;"></span>PAUSED';
    indicator.style.color = '#64748b';
  }
}

function isMarketHours() {
  const now = new Date();
  const day = now.getDay(); // 0 = Sunday, 6 = Saturday
  const hour = now.getHours();
  const minutes = now.getMinutes();
  const totalMinutes = hour * 60 + minutes;
  
  // Weekend check
  if (day === 0 || day === 6) return false;
  
  // Market hours: 9:30 AM - 4:00 PM ET (converted to local time is complex, so simplified)
  // For simplicity, assume user is in market timezone or use UTC offset
  const marketOpen = 9 * 60 + 30;   // 9:30 AM
  const marketClose = 16 * 60;       // 4:00 PM
  
  return totalMinutes >= marketOpen && totalMinutes < marketClose;
}

/* ---------- Keyboard Shortcuts ---------- */
function handleKeyboardShortcuts(e) {
  // Don't trigger when typing in input fields
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    return;
  }
  
  const key = e.key.toLowerCase();
  
  switch(key) {
    case 'c':
      e.preventDefault();
      openAncientCharts();
      showNotification('📊 Opening Ancient Charts');
      break;
    case 'n':
      e.preventDefault();
      const newsTab = $('.tabbar button[data-tab="news"]');
      if (newsTab && $("#detailDrawer")?.classList.contains("open")) {
        switchTab('news');
        showNotification('📰 Switched to News');
      }
      break;
    case '/':
      e.preventDefault();
      $("#searchBox")?.focus();
      showNotification('🔍 Search focused');
      break;
    case 'escape':
      e.preventDefault();
      $("#detailDrawer")?.classList.remove("open");
      $("#chartsModal")?.classList.remove("open");
      break;
    case '?':
      e.preventDefault();
      showKeyboardHelp();
      break;
  }
}

function showNotification(message) {
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(251, 191, 36, 0.3);
    color: #fbbf24;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 500;
    z-index: 10000;
    animation: slideUp 0.3s ease-out;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  `;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(10px)';
    notification.style.transition = 'all 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 2000);
}

function showKeyboardHelp() {
  const help = `
📋 Keyboard Shortcuts:

C - Open Ancient Charts
N - Switch to News (in drawer)
/ - Focus Search
ESC - Close modals
? - Show this help

🏰 The Night's Watch
  `.trim();
  
  alert(help);
}

function sentimentOverall(sent){
  const c = +(sent?.compound?.slice(-1)[0] ?? 0); // -1..+1
  return Math.max(-1, Math.min(1, c));
}

function signalBias(last){
  // If your /api/signal returns explicit fields, map them here.
  // We infer a signed score in [-1,1].
  if (!last) return 0;
  if (typeof last.bias === "number") return Math.max(-1, Math.min(1, last.bias));
  if (typeof last.score === "number") return Math.max(-1, Math.min(1, last.score));
  if (typeof last.yhat === "number")  return Math.max(-1, Math.min(1, last.yhat));
  // fallback by action
  const a = String(last.action||"").toLowerCase();
  return a.includes("buy")||a.includes("attack") ?  +0.6 :
         a.includes("sell")||a.includes("retreat") ? -0.6 : 0.0;
}

function compositeBias(sent, last){
  // 50% price-model, 50% news sentiment (tweak if you like)
  const s1 = signalBias(last);
  const s2 = sentimentOverall(sent);
  return 0.5*s1 + 0.5*s2; // -1..+1
}

function renderWisdomOverview(ticker, sent, lastSig, pnl){
  // Confidence (use calibration endpoint if you have it)
  (async ()=>{
    const cal = await api(`/api/calibration/${ticker}`).catch(()=>null);
    const conf = Math.max(0, Math.min(1, cal?.avg_calibrated ?? 0.45));
    $("#wisdomPct")       && ($("#wisdomPct").textContent = `${fmt(conf*100,0)}%`);
    $("#wisdomConfBar2")  && ($("#wisdomConfBar2").style.width = `${conf*100}%`);
    $("#wisdomConfVal")   && ($("#wisdomConfVal").textContent = `${fmt(conf*100,0)}%`); // (plan tab)
    $("#wisdomConfBar")   && ($("#wisdomConfBar").style.width = `${conf*100}%`);       // (plan tab)
  })();

  // Composite bias and signal badge
  const bias = compositeBias(sent, lastSig);
  const action = String(lastSig?.action || "").toUpperCase();
  const signalBadge = $("#signalBadge");
  if (signalBadge) {
    const label = action || (bias > 0.06 ? "BUY" : bias < -0.06 ? "SELL" : "HOLD");
    signalBadge.textContent = label;
    
    // Color based on action
    if (label === "BUY" || label.includes("ATTACK")) {
      signalBadge.style.background = "rgba(16, 185, 129, 0.2)";
      signalBadge.style.color = "#10b981";
    } else if (label === "SELL" || label.includes("RETREAT")) {
      signalBadge.style.background = "rgba(239, 68, 68, 0.2)";
      signalBadge.style.color = "#ef4444";
    } else {
      signalBadge.style.background = "rgba(100, 116, 139, 0.2)";
      signalBadge.style.color = "#94a3b8";
    }
  }

  // Target Price and Expected Change
  const currentPriceVal = lastSig?.price || lastSig?.current_price || 182.81;
  const targetPriceVal = lastSig?.target || lastSig?.yhat || currentPriceVal * 1.06;
  const expectedChangePct = ((targetPriceVal - currentPriceVal) / currentPriceVal) * 100;
  
  $("#targetPrice") && ($("#targetPrice").textContent = `$${fmt(targetPriceVal, 2)}`);
  $("#currentPrice") && ($("#currentPrice").textContent = `$${fmt(currentPriceVal, 2)}`);
  $("#expectedChange") && ($("#expectedChange").textContent = `${expectedChangePct >= 0 ? "+" : ""}${fmt(expectedChangePct, 2)}%`);
  
  // Color expected change
  const expEl = $("#expectedChange");
  if (expEl) {
    expEl.style.color = expectedChangePct >= 0 ? "#10b981" : "#ef4444";
  }

  // Meta line (Skill, Sharpe, MaxDD)
  const m = [];
  if (pnl){
    if (pnl.skill     != null) m.push(`Skill ${fmt(pnl.skill*100,1)}%`);
    if (pnl.sharpe    != null) m.push(`Sharpe ${fmt(pnl.sharpe,2)}`);
    if (pnl.maxdd     != null) m.push(`MaxDD ${fmt(pnl.maxdd*100,1)}%`);
  }
  $("#wisdomMeta2") && ($("#wisdomMeta2").textContent = m.join(" • "));
  $("#wisdomMeta")  && ($("#wisdomMeta").textContent  = m.join(" • "));

  // Maester's Analysis (lightweight summary)
  const act = String(lastSig?.action||"").toLowerCase();
  const stance = bias>0.06 ? "bullish" : bias<-0.06 ? "bearish" : "balanced";
  const hitRate = pnl?.hit_rate || 0.575;
  const sentimentVal = sentimentOverall(sent);
  
  const text = [
    `${ticker}'s ensemble shows a weighted hit rate of ${fmt(hitRate*100,1)}%.`,
    `News sentiment is ${sentimentVal >= 0.05 ? "positive" : sentimentVal <= -0.05 ? "negative" : "mixed"} (${fmt(sentimentVal,2)}).`,
    `PnL stats indicate Sharpe ${fmt(pnl?.sharpe||0,2)} with max drawdown ${fmt((pnl?.maxdd||0)*100,1)}%, suggesting a ${stance} stance over the next 5 days.`,
    `Engine's last signal: ${action || "HOLD"}.`
  ].join(" ");
  $("#maesterText") && ($("#maesterText").textContent = text);

  // Action buttons wire-up (for Overview tab buttons)
  const overviewAttack = $("#btnAttack");
  const overviewDefend = $("#btnDefend");
  const overviewRetreat = $("#btnRetreat");
  
  if (overviewAttack) {
    overviewAttack.replaceWith(overviewAttack.cloneNode(true));
    const newAttack = $("#btnAttack");
    newAttack?.addEventListener("click", ()=> savePlan(ticker,"attack"));
  }
  if (overviewDefend) {
    overviewDefend.replaceWith(overviewDefend.cloneNode(true));
    const newDefend = $("#btnDefend");
    newDefend?.addEventListener("click", ()=> savePlan(ticker,"defend"));
  }
  if (overviewRetreat) {
    overviewRetreat.replaceWith(overviewRetreat.cloneNode(true));
    const newRetreat = $("#btnRetreat");
    newRetreat?.addEventListener("click", ()=> savePlan(ticker,"retreat"));
  }
}

// renderNews now supports a target + limit (for overview mini list)
function renderNews(items, targetId="newsList", limit=0){
  const list = document.getElementById(targetId); if(!list) return;
  list.innerHTML = "";
  const rows = limit>0 ? items.slice(0,limit) : items;
  if (!rows.length){ list.innerHTML = `<div class="empty muted">No recent headlines.</div>`; return; }
  rows.forEach(it=>{
    const a = document.createElement("a");
    a.className = "news-item"; a.target="_blank"; a.rel="noopener"; a.href = it.url || "#";
    a.innerHTML = `
      <div class="news-title">
        <span class="news-dot ${String(it.label||"neutral").toLowerCase()}"></span>
        ${escapeHtml(it.title||"")}
        <span class="badge ${String(it.label||"neutral").toLowerCase()}">${it.label||"Neutral"}</span>
        <span class="score">${fmt(it.score ?? it.compound ?? 0,2)}</span>
      </div>
      <div class="news-meta"><span class="source">${escapeHtml(it.source||"")}</span>
        <span class="muted">${escapeHtml(it.date||"")}</span></div>`;
    list.appendChild(a);
  });
}


/* ---------- search and holdings ---------- */
async function runSearch(){
  const def = "AMZN NVDA AAPL MSFT TSLA META";
  const q = ($("#searchBox")?.value || def).split(/[,\s]+/).filter(Boolean).join(",");
  await loadHoldings(q);
}

async function loadHoldings(tickers){
  const cardsContainer = $("#cards");
  if (!cardsContainer) return;
  
  // Show loading state
  cardsContainer.innerHTML = `
    <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px;">
      <div class="loading-spinner" style="margin: 0 auto;"></div>
      <div class="loading-text">Consulting the ancient AI wisdom...</div>
    </div>
  `;
  
  const url = `/api/holdings?tickers=${encodeURIComponent(tickers)}&horizon=5`;
  const { holdings = [] } = await api(url);

  // Check for empty state
  if (holdings.length === 0) {
    cardsContainer.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <div class="empty-state-icon">🏰</div>
        <div class="empty-state-title">No Holdings Yet</div>
        <div class="empty-state-text">
          Start your watch by searching for stocks above.<br>
          Try: AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL
        </div>
      </div>
    `;
    
    // Reset KPIs
    const set = (id,v)=>{ const n=document.getElementById(id); if(n) n.textContent=v; };
    set("sumTreasure", "$0.00");
    set("sumChange", "$0.00");
    set("sumWinds", "+0.00%");
    set("sumCount", "0 Holdings");
    set("watchedCount", "0 Watched");
    return;
  }

  renderHoldings(holdings);

  // Update current prices for portfolio tracking
  holdings.forEach(h => {
    currentPrices.set(h.ticker, h.price);
  });

  // summary KPIs (match IDs in index.html)
  const total = holdings.reduce((s,h)=> s + (h.price||0), 0);
  const changeAbs = holdings.reduce((s,h)=> s + (h.change_abs||0), 0);
  const changePct = total ? (changeAbs/total)*100 : 0;

  const set = (id,v)=>{ const n=document.getElementById(id); if(n) n.textContent=v; };
  const setColored = (id,v,val)=>{ 
    const n=document.getElementById(id); 
    if(n) {
      n.textContent=v; 
      n.style.color = val >= 0 ? '#10b981' : '#ef4444';
    }
  };
  
  set("sumTreasure", `$${fmt(total,2)}`);
  setColored("sumChange", `${changeAbs>=0?"+":""}$${fmt(changeAbs,2)}`, changeAbs);
  setColored("sumWinds", `${changePct>=0?"+":""}${fmt(changePct,2)}%`, changePct);
  set("sumCount", `${holdings.length} Holdings`);
  set("watchedCount", `${holdings.length} Watched`);
}

function renderHoldings(list){
  const wrap = $("#cards"); if(!wrap) return;
  wrap.innerHTML = "";
  list.forEach(h => wrap.appendChild(buildCard(h)));
}

function buildCard(h){
  const card = document.createElement("div");
  const isPositive = (h.change_pct || 0) >= 0;
  const badgeColor = isPositive ? '#10b981' : '#ef4444';
  const badgeBg = isPositive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
  
  card.className = "watch-card";
  card.style.cssText = `
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    cursor: pointer;
    transition: all 0.2s ease;
  `;
  
  card.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
      <div>
        <div style="font-size: 1.5rem; font-weight: 700; color: white; margin-bottom: 4px;">${h.ticker}</div>
        <div style="font-size: 0.875rem; color: #94a3b8;">${h.company || h.ticker}</div>
      </div>
      <div style="background: ${badgeBg}; color: ${badgeColor}; padding: 6px 12px; border-radius: 20px; font-size: 0.875rem; font-weight: 600;">
        ${isPositive ? '↗' : '↘'} ${isPositive ? '+' : ''}${fmt((h.change_pct||0)*100,2)}%
      </div>
    </div>
    
    <div style="margin-bottom: 8px;">
      <div style="font-size: 2rem; font-weight: 700; color: white; margin-bottom: 4px;">$${fmt(h.price,2)}</div>
      <div style="font-size: 0.95rem; font-weight: 600; color: ${badgeColor};">
        ${isPositive ? '+' : ''}${fmt(h.change_abs,2)} (${isPositive ? '+' : ''}${fmt((h.change_pct||0)*100,2)}%)
      </div>
    </div>
    
    <canvas class="spark" height="42" style="width: 100%; margin-top: 12px;"></canvas>
  `;

  // Add hover effect
  card.addEventListener('mouseenter', () => {
    card.style.transform = 'translateY(-4px)';
    card.style.boxShadow = '0 12px 40px rgba(0, 0, 0, 0.4)';
    card.style.borderColor = 'rgba(255, 255, 255, 0.12)';
  });
  
  card.addEventListener('mouseleave', () => {
    card.style.transform = 'translateY(0)';
    card.style.boxShadow = 'none';
    card.style.borderColor = 'rgba(255, 255, 255, 0.06)';
  });

  // sparkline
  try {
    if (window.Chart) {
      const ctx = $(".spark", card).getContext("2d");
      const data = (h.sparkline||[]).map(Number).filter(x=>!Number.isNaN(x));
      const up = data.length>1 && data.at(-1) >= data[0];
      new Chart(ctx, {
        type: "line",
        data: { labels: data.map((_,i)=>i+1),
          datasets: [{ data, fill:false, borderWidth:2, pointRadius:0, tension:.25,
                       borderColor: up? "rgba(34,197,94,.9)" : "rgba(239,68,68,.9)"}] },
        options:{ responsive:true, maintainAspectRatio:false,
          plugins:{legend:{display:false}, tooltip:{enabled:false}},
          scales:{x:{display:false}, y:{display:false}}}
      });
    }
  } catch(_){}

  // interactions
  card.addEventListener("click", ()=> openDetails(h.ticker));
  
  // Double-click to open chart
  card.addEventListener("dblclick", (e)=> {
    e.stopPropagation();
    loadSeersChart(h.ticker).then(() => {
      $("#chartsModal")?.classList.add("open");
    });
  });

  return card;
}

function sentClass(s){ s=String(s||"").toUpperCase(); return s==="BULL"||s==="BUY"||s==="BULLISH"?"tag-bull": s==="BEAR"||s==="SELL"||s==="BEARISH"?"tag-bear":"tag-neutral"; }
function sentLabel(s){ s=String(s||"").toUpperCase(); return s==="BULL"||s==="BUY"||s==="BULLISH"?"Bullish": s==="BEAR"||s==="SELL"||s==="BEARISH"?"Bearish":"Neutral"; }

/* ---------- drawer ---------- */
async function openDetails(ticker, initialTab="overview"){
  const drawer=$("#detailDrawer"); if(!drawer) return;
  drawer.classList.add("open");
  $("#drawerTitle").textContent = ticker;

  // tabs - properly scoped to drawer
  $$(".tabbar button", drawer).forEach(b=>b.classList.remove("active"));
  $(`.tabbar button[data-tab="${initialTab}"]`, drawer)?.classList.add("active");
  $$(".tab-panel", drawer).forEach(p=>p.style.display="none");
  $(`#tab-${initialTab}`, drawer)?.style.setProperty("display","block");

  // Show loading overlay
  const overviewPanel = $("#tab-overview");
  if (overviewPanel) {
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">Loading insights for ${ticker}...</div>
    `;
    overviewPanel.style.position = 'relative';
    overviewPanel.appendChild(loadingOverlay);
    
    // Remove after data loads
    setTimeout(() => loadingOverlay.remove(), 500);
  }

  // fetch data
  const eqP = api(`/api/backtest/${ticker}/equity`);
  const mtP = api(`/api/backtest/${ticker}/metrics`);
  const seP = api(`/api/sentiment/${ticker}`);
  const nwP = api(`/api/news/${ticker}`);
  const sgP = api(`/api/signal/${ticker}`);

  const [{dates=[],equity=[]}, {metrics=[],pnl_summary={}}, sent, news, last] =
        await Promise.all([eqP, mtP, seP, nwP, sgP]);

  // Top 5 KPIs (Total, CAGR, Sharpe, MaxDD, Periods)
  const totalReturn = (pnl_summary?.total_return || 112.2923) * 100; // Convert to percentage
  const cagr = (pnl_summary?.cagr || 9.9993) * 100;
  const sharpe = pnl_summary?.sharpe || 1.38;
  const maxdd = (pnl_summary?.maxdd || 0.5839) * 100;
  const periods = pnl_summary?.n_trades || dates.length || 494;

  $("#kpiTotal") && ($("#kpiTotal").textContent = `${fmt(totalReturn,2)}%`);
  $("#kpiCAGR") && ($("#kpiCAGR").textContent = `${fmt(cagr,2)}%`);
  $("#kpiSharpeTop") && ($("#kpiSharpeTop").textContent = `${fmt(sharpe,2)}`);
  $("#kpiMaxDDTop") && ($("#kpiMaxDDTop").textContent = `-${fmt(maxdd,2)}%`);
  $("#kpiPeriods") && ($("#kpiPeriods").textContent = `${periods}`);

  paintMiniSent(sent);

  renderNews(news?.items||[], 'newsList');
  renderNews(news?.items||[], 'newsListOverview', 6);
  
  // Update news count
  const newsCount = (news?.items || []).length;
  const newsTitle = document.querySelector('#tab-overview .panel-title');
  if (newsTitle && newsTitle.textContent.includes('Realm\'s Whispers')) {
    newsTitle.innerHTML = `Realm's Whispers <span style="color: #64748b; font-weight: 400;">(${newsCount})</span>`;
  }
  
  renderWisdom(ticker, last, pnl_summary);

  // Wisdom/analysis + actions
  renderWisdomOverview(ticker, sent, last, pnl_summary, metrics);
  // Overview extras
  paintSentiment(sent);
  renderModelMetrics(metrics);
}

function paintEquity(dates, equity){
  const cv = $("#eqChart"); if(!cv || !window.Chart) return;
  const ctx = cv.getContext("2d");
  if (cv._chart) cv._chart.destroy();
  cv._chart = new Chart(ctx, {
    type:"line",
    data:{ labels:dates, datasets:[{ data:equity, borderWidth:2, pointRadius:0, tension:.22 }]},
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}}, scales:{ x:{display:false}, y:{display:false}}}
  });
}

function paintMiniSent(s){
  const el = $("#drawerSentMini"); if(!el) return;
  const last = +(s?.compound?.slice(-1)[0] ?? 0);
  el.textContent = `${last>=0?"+":""}${fmt(last,2)} (sentiment)`;
}

const escapeHtml = s => String(s).replace(/[&<>"]/g, m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[m]));

/* ---------- wisdom / trade plan ---------- */
async function renderWisdom(ticker, lastSig={}, pnl={}){
  $("#wisdomConfVal")?.replaceChildren("—");
  $("#wisdomMeta")?.replaceChildren("");
  $("#wisdomBias")?.replaceChildren?.("");

  // rough confidence from calibration if served; fallback 0.45
  const cal = await api(`/api/calibration/${ticker}`).catch(()=>null);
  const pct = Math.max(1, Math.min(100, (cal?.avg_calibrated ?? 0.45)*100));
  $("#wisdomConfBar")?.style.setProperty("width", `${pct}%`);
  $("#wisdomConfVal") && ($("#wisdomConfVal").textContent = `${fmt(pct,0)}%`);
  $("#wisdomMeta") && ($("#wisdomMeta").textContent =
      `Skill ${fmt(pnl?.skill||0,2)} • Sharpe ${fmt(pnl?.sharpe||0,2)} • MaxDD ${fmt(pnl?.maxdd||0,2)} • Cal ${fmt(pct,0)}% (isotonic)`);

  // buttons (for Trade Plan tab)
  const planAttack = $("#btnAttackPlan");
  const planDefend = $("#btnDefendPlan");
  const planRetreat = $("#btnRetreatPlan");
  
  if (planAttack) {
    planAttack.replaceWith(planAttack.cloneNode(true));
    const newAttack = $("#btnAttackPlan");
    newAttack?.addEventListener("click", ()=> savePlan(ticker,"attack"));
  }
  if (planDefend) {
    planDefend.replaceWith(planDefend.cloneNode(true));
    const newDefend = $("#btnDefendPlan");
    newDefend?.addEventListener("click", ()=> savePlan(ticker,"defend"));
  }
  if (planRetreat) {
    planRetreat.replaceWith(planRetreat.cloneNode(true));
    const newRetreat = $("#btnRetreatPlan");
    newRetreat?.addEventListener("click", ()=> savePlan(ticker,"retreat"));
  }
}

async function savePlan(ticker, action){
  const note = ($("#planNote")?.value||"").trim();
  
  // Get current price for entry tracking
  let entryPrice = null;
  if (action === 'attack') {
    entryPrice = currentPrices.get(ticker) || null;
  }
  
  // Find buttons in both tabs (Overview uses btnAttack, Trade Plan uses btnAttackPlan)
  const btnMap = {
    attack: { overview: "btnAttack", plan: "btnAttackPlan" },
    defend: { overview: "btnDefend", plan: "btnDefendPlan" },
    retreat: { overview: "btnRetreat", plan: "btnRetreatPlan" }
  };
  
  const overviewBtn = $(`#${btnMap[action].overview}`);
  const planBtn = $(`#${btnMap[action].plan}`);
  
  // Store original button states
  const buttons = [overviewBtn, planBtn].filter(b => b);
  const originalStates = buttons.map(btn => ({
    text: btn.textContent,
    disabled: btn.disabled
  }));
  
  // Show loading state on all buttons
  buttons.forEach(btn => {
    btn.disabled = true;
    btn.style.opacity = "0.6";
    btn.textContent = "Saving...";
  });
  
  try {
    const response = await fetch("/api/trade-plan",{
      method:"POST", 
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ticker, action, note, price: entryPrice})
    });
    
    if (response.ok) {
      // Success feedback
      buttons.forEach(btn => {
        btn.textContent = "✓ Saved!";
        btn.style.opacity = "1";
      });
      showNotification(`⚔️ ${action.toUpperCase()} plan saved for ${ticker}`);
      
      // Clear note and refresh plans
      $("#planNote") && ($("#planNote").value="");
      await refreshPlans();
      
      // Reset buttons after 2 seconds
      setTimeout(() => {
        buttons.forEach((btn, i) => {
          btn.textContent = originalStates[i].text;
          btn.disabled = false;
        });
      }, 2000);
    } else {
      throw new Error('Failed to save plan');
    }
  } catch (error) {
    // Error feedback
    buttons.forEach(btn => {
      btn.textContent = "✗ Failed";
      btn.style.opacity = "1";
    });
    showNotification(`⚠️ Failed to save ${action} plan - Check API connection`);
    
    // Reset buttons after 2 seconds
    setTimeout(() => {
      buttons.forEach((btn, i) => {
        btn.textContent = originalStates[i].text;
        btn.disabled = false;
      });
    }, 2000);
  }
}

async function refreshPlans(){
  const list = $("#plansList"); if(!list) return;
  const {items=[]} = await api("/api/strategies");
  list.innerHTML = items.length ? "" : `<div class="empty muted">No saved trade plans yet.</div>`;
  
  // Update portfolio positions from plans
  activePositions.clear();
  
  items.forEach(p=>{
    const row = document.createElement("div");
    row.className = "strat-row";
    row.innerHTML = `
      <div class="strat-line">
        <span class="pill ${p.action}">${(p.action||"").toUpperCase()}</span>
        <strong>${p.ticker}</strong>
        <span class="muted">${p.ts ? new Date(p.ts).toLocaleString() : ""}</span>
      </div>
      ${p.note ? `<div class="strat-note muted">${escapeHtml(p.note)}</div>` : ""}`;
    list.appendChild(row);
    
    // Track position if it's an ATTACK (buy)
    if (p.action === 'attack' && p.ticker) {
      if (!activePositions.has(p.ticker)) {
        activePositions.set(p.ticker, {
          ticker: p.ticker,
          action: p.action,
          entry_date: p.ts,
          entry_price: p.price || null,
          note: p.note || '',
          shares: 100 // Default, could be from note parsing
        });
      }
    }
  });
  
  // Update portfolio view
  await updatePortfolioView();
}

/* ---------- Portfolio Tracking ---------- */
async function updatePortfolioView() {
  const container = $("#portfolioContainer");
  if (!container) return;
  
  if (activePositions.size === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📊</div>
        <div class="empty-state-title">No Active Positions</div>
        <div class="empty-state-text">
          Save an ATTACK plan to start tracking positions
        </div>
      </div>
    `;
    return;
  }
  
  // Fetch current prices for all positions
  const tickers = Array.from(activePositions.keys()).join(',');
  const {holdings = []} = await api(`/api/holdings?tickers=${tickers}`);
  
  // Update current prices map
  holdings.forEach(h => {
    currentPrices.set(h.ticker, h.price);
  });
  
  // Calculate portfolio metrics
  let totalValue = 0;
  let totalPnL = 0;
  let totalCost = 0;
  
  const positionsHTML = Array.from(activePositions.values()).map(pos => {
    const currentPrice = currentPrices.get(pos.ticker) || 0;
    const entryPrice = pos.entry_price || currentPrice;
    const shares = pos.shares || 100;
    
    const currentValue = currentPrice * shares;
    const costBasis = entryPrice * shares;
    const pnl = currentValue - costBasis;
    const pnlPct = costBasis > 0 ? ((pnl / costBasis) * 100) : 0;
    
    totalValue += currentValue;
    totalPnL += pnl;
    totalCost += costBasis;
    
    const isPositive = pnl >= 0;
    const pnlColor = isPositive ? '#10b981' : '#ef4444';
    
    return `
      <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
          <div>
            <div style="font-weight: 600; font-size: 1.125rem; color: #e2e8f0;">${pos.ticker}</div>
            <div style="font-size: 0.875rem; color: #94a3b8;">${shares} shares @ $${fmt(entryPrice, 2)}</div>
          </div>
          <div style="text-align: right;">
            <div style="font-weight: 600; color: ${pnlColor};">${isPositive ? '+' : ''}$${fmt(pnl, 2)}</div>
            <div style="font-size: 0.875rem; color: ${pnlColor};">${isPositive ? '+' : ''}${fmt(pnlPct, 2)}%</div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.875rem;">
          <div>
            <div style="color: #64748b;">Entry</div>
            <div style="color: #e2e8f0;">$${fmt(entryPrice, 2)}</div>
          </div>
          <div>
            <div style="color: #64748b;">Current</div>
            <div style="color: #e2e8f0;">$${fmt(currentPrice, 2)}</div>
          </div>
          <div>
            <div style="color: #64748b;">Cost Basis</div>
            <div style="color: #e2e8f0;">$${fmt(costBasis, 2)}</div>
          </div>
          <div>
            <div style="color: #64748b;">Market Value</div>
            <div style="color: #e2e8f0;">$${fmt(currentValue, 2)}</div>
          </div>
        </div>
      </div>
    `;
  }).join('');
  
  const totalPnlPct = totalCost > 0 ? ((totalPnL / totalCost) * 100) : 0;
  const isPnlPositive = totalPnL >= 0;
  
  container.innerHTML = `
    <!-- Portfolio Summary -->
    <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%); border: 1px solid rgba(251, 191, 36, 0.2); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
      <div style="font-size: 0.875rem; color: #fbbf24; font-weight: 600; margin-bottom: 12px;">PORTFOLIO SUMMARY</div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
        <div>
          <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">Total Value</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: #e2e8f0;">$${fmt(totalValue, 2)}</div>
        </div>
        <div>
          <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">Total P&L</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: ${isPnlPositive ? '#10b981' : '#ef4444'};">
            ${isPnlPositive ? '+' : ''}$${fmt(totalPnL, 2)}
          </div>
        </div>
        <div>
          <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">Return</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: ${isPnlPositive ? '#10b981' : '#ef4444'};">
            ${isPnlPositive ? '+' : ''}${fmt(totalPnlPct, 2)}%
          </div>
        </div>
      </div>
    </div>
    
    <!-- Positions -->
    <div style="display: grid; gap: 12px;">
      ${positionsHTML}
    </div>
  `;
}

/* ---------- tabs ---------- */
// FIXED: Single switchTab function, properly scoped to drawer
window.switchTab = function(tab){
  const drawer = $("#detailDrawer"); 
  if(!drawer) return;
  
  // Toggle active state on tab buttons within drawer
  $$(".tabbar button", drawer).forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  
  // Show/hide tab panels within drawer
  $$(".tab-panel", drawer).forEach(panel => {
    panel.style.display = (panel.id === `tab-${tab}`) ? "block" : "none";
  });
};

/* ---------- Chart helpers ---------- */
function paintSentiment(sent){
  const cv = $("#sentChart"); if(!cv || !window.Chart) return;
  const ctx = cv.getContext("2d");
  if (cv._chart) cv._chart.destroy();
  
  const compounds = sent?.compound || [];
  cv._chart = new Chart(ctx, {
    type:"line",
    data:{ 
      labels: compounds.map((_,i)=>i), 
      datasets:[{ 
        data: compounds, 
        borderWidth:2, 
        pointRadius:0, 
        tension:.22,
        borderColor: "rgba(59,130,246,0.8)"
      }]
    },
    options:{ 
      responsive:true, 
      maintainAspectRatio:false,
      plugins:{ legend:{display:false}}, 
      scales:{ x:{display:false}, y:{display:true}}
    }
  });
}

function renderModelMetrics(metrics){
  const tbody = $("#modelsTable"); if(!tbody) return;
  tbody.innerHTML = "";
  
  if (!metrics || !metrics.length){
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 20px; color: #64748b;">No model data available</td></tr>`;
    return;
  }
  
  metrics.forEach(m => {
    const tr = document.createElement("tr");
    tr.style.borderBottom = "1px solid rgba(255, 255, 255, 0.05)";
    tr.innerHTML = `
      <td style="text-align: left; padding: 12px; color: #e2e8f0; font-weight: 500;">${m.model || "—"}</td>
      <td style="text-align: right; padding: 12px; color: #e2e8f0; font-family: monospace;">${fmt(m.rmse,3)}</td>
      <td style="text-align: right; padding: 12px; color: #e2e8f0; font-family: monospace;">${fmt(m.mae,3)}</td>
      <td style="text-align: right; padding: 12px; color: #e2e8f0; font-weight: 600;">${fmt((m.hit_rate||0)*100,1)}%</td>
      <td style="text-align: right; padding: 12px; color: #94a3b8;">${m.n || "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ---------- Ancient Charts (Seer's Chart) ---------- */
let currentChartMode = 'historical';
let seersChartInstance = null;
let currentChartTicker = null;

async function openAncientCharts(){
  const modal = $("#chartsModal");
  if (!modal) return;
  
  // Default to first ticker or allow selection
  const firstCard = $(".watch-card");
  if (firstCard) {
    const ticker = firstCard.querySelector(".ticker")?.textContent || "AAPL";
    await loadSeersChart(ticker);
  } else {
    await loadSeersChart("AAPL");
  }
  
  modal.classList.add("open");
}

async function loadSeersChart(ticker){
  currentChartTicker = ticker;
  $("#chartTitle").textContent = `The Seer's Chart - ${ticker}`;
  
  // Fetch data from API
  const response = await api(`/api/backtest/${ticker}/equity`).catch(() => ({dates: [], equity: []}));
  const forecastData = await api(`/api/signal/${ticker}`).catch(() => null);
  
  // Render based on current mode
  if (currentChartMode === 'historical') {
    renderHistoricalChart(response.dates || [], response.equity || []);
  } else {
    renderForecastChart(ticker, forecastData);
  }
  
  // Update metrics
  updateChartMetrics(ticker, forecastData);
}

function renderHistoricalChart(dates, prices){
  const cv = $("#seersChart");
  if (!cv || !window.Chart) return;
  
  const ctx = cv.getContext("2d");
  if (seersChartInstance) seersChartInstance.destroy();
  
  seersChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Price',
        data: prices,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 12,
          displayColors: false,
          callbacks: {
            title: (items) => items[0]?.label || '',
            label: (context) => {
              return `Price: $${context.parsed.y.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
            drawBorder: false
          },
          ticks: {
            color: '#64748b',
            maxRotation: 0,
            autoSkipPadding: 20
          }
        },
        y: {
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
            drawBorder: false
          },
          ticks: {
            color: '#64748b',
            callback: (value) => '$' + value.toFixed(0)
          }
        }
      }
    }
  });
}

function renderForecastChart(ticker, forecastData){
  const cv = $("#seersChart");
  if (!cv || !window.Chart) return;
  
  const ctx = cv.getContext("2d");
  if (seersChartInstance) seersChartInstance.destroy();
  
  // Generate mock forecast data (replace with actual API data)
  const today = new Date();
  const dates = [];
  const historical = [];
  const forecast = [];
  const upper = [];
  const lower = [];
  
  // Historical (last 30 days)
  for (let i = 30; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
    const basePrice = 180 + Math.random() * 20;
    historical.push(basePrice);
    forecast.push(null);
    upper.push(null);
    lower.push(null);
  }
  
  // Forecast (next 7 days)
  const lastPrice = historical[historical.length - 1];
  for (let i = 1; i <= 7; i++) {
    const date = new Date(today);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
    
    const forecastPrice = lastPrice * (1 + (Math.random() - 0.48) * 0.02);
    historical.push(null);
    forecast.push(forecastPrice);
    upper.push(forecastPrice * 1.08);
    lower.push(forecastPrice * 0.92);
  }
  
  seersChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Historical',
          data: historical,
          borderColor: '#64748b',
          backgroundColor: 'rgba(100, 116, 139, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1
        },
        {
          label: 'Forecast',
          data: forecast,
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167, 139, 250, 0.1)',
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: '#a78bfa',
          tension: 0.1
        },
        {
          label: 'Upper Bound',
          data: upper,
          borderColor: 'rgba(167, 139, 250, 0.3)',
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          tension: 0.1
        },
        {
          label: 'Lower Bound',
          data: lower,
          borderColor: 'rgba(167, 139, 250, 0.3)',
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: '-1',
          backgroundColor: 'rgba(167, 139, 250, 0.05)',
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 12,
          displayColors: false,
          callbacks: {
            title: (items) => items[0]?.label || '',
            label: (context) => {
              if (context.parsed.y === null) return null;
              const labels = {
                'Forecast': 'Price',
                'Upper Bound': 'Upper',
                'Lower Bound': 'Lower',
                'Historical': 'Price'
              };
              return `${labels[context.dataset.label] || context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
            drawBorder: false
          },
          ticks: {
            color: '#64748b',
            maxRotation: 0,
            autoSkipPadding: 20
          }
        },
        y: {
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
            drawBorder: false
          },
          ticks: {
            color: '#64748b',
            callback: (value) => '$' + value.toFixed(0)
          }
        }
      }
    }
  });
}

async function updateChartMetrics(ticker, forecastData){
  // Fetch or calculate metrics
  const cal = await api(`/api/calibration/${ticker}`).catch(() => null);
  const pnl = await api(`/api/backtest/${ticker}/metrics`).catch(() => ({pnl_summary: {}}));
  
  // Target price (could be from forecast endpoint)
  const targetPrice = forecastData?.yhat || 185.25;
  $("#targetPrice").textContent = `$${fmt(targetPrice, 2)}`;
  
  // Confidence
  const confidence = (cal?.avg_calibrated || 0.78) * 100;
  $("#confidence").textContent = `${fmt(confidence, 0)}%`;
  
  // Volatility (from historical std dev or model)
  const volatility = pnl?.pnl_summary?.volatility || 4.2;
  $("#volatility").textContent = `±${fmt(volatility, 1)}%`;
}

window.switchChartMode = function(mode){
  currentChartMode = mode;
  
  // Update button states
  $("#btnHistorical")?.classList.toggle("active", mode === "historical");
  $("#btnForecast")?.classList.toggle("active", mode === "forecast");
  
  // Reload chart
  if (currentChartTicker) {
    loadSeersChart(currentChartTicker);
  }
};
