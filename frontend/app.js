document.addEventListener('DOMContentLoaded', () => {
    // --- STATE & CONFIG ---
    const API_BASE_URL = 'http://127.0.0.1:8000';
    const WS_URL = 'ws://127.0.0.1:8000/ws/metrics';
    let trafficChart;
    const CHART_MAX_POINTS = 60;

    // --- DOM ELEMENTS ---
    const elements = {
        statusDot: document.getElementById('status-dot'),
        statusText: document.getElementById('status-text'),
        totalReq60s: document.getElementById('total-req-60s'),
        totalReq10s: document.getElementById('total-req-10s'),
        activeIps: document.getElementById('active-ips'),
        ddosAttacks: document.getElementById('ddos-attacks'),
        xssAttacks: document.getElementById('xss-attacks'),
        sqlAttacks: document.getElementById('sql-attacks'),
        topTalkersBody: document.getElementById('top-talkers-body'),
        blockedIpsBody: document.getElementById('blocked-ips-body'),
        eventsContainer: document.getElementById('events-container'),
        trafficChartCanvas: document.getElementById('traffic-chart'),
    };

    // --- CHART ---
    function initChart() {
        if (!window.Chart) {
            console.error("Chart.js is not loaded!");
            return;
        }
        const ctx = elements.trafficChartCanvas.getContext('2d');
        trafficChart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [{
                    label: 'req/s', 
                    data: [], 
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.2)',
                    fill: true, 
                    tension: 0.4, 
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    x: { 
                        grid: { display: false },
                        ticks: { display: false } 
                    },
                    y: { 
                        beginAtZero: true, 
                        grid: { color: '#30363d' }, 
                        ticks: { color: '#8b949e' } 
                    }
                },
                plugins: { 
                    legend: { display: false } 
                }
            }
        });
    }

    // --- UI UPDATE LOGIC ---
    const asNum = (v, fallback = 0) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
    };

    function updateUI(data) {
        const total60 = asNum(data.total_req_rate_60s);
        const total10 = asNum(data.total_req_rate_10s);
        const active = asNum(data.active_ips_count);

        elements.totalReq60s.textContent = total60.toFixed(2);
        elements.totalReq10s.textContent = total10.toFixed(2);
        elements.activeIps.textContent = active;

        // Update attack statistics
        if (data.attack_stats) {
            elements.ddosAttacks.textContent = asNum(data.attack_stats.ddos);
            elements.xssAttacks.textContent = asNum(data.attack_stats.xss);
            elements.sqlAttacks.textContent = asNum(data.attack_stats.sql);
        }

        updateTable(elements.topTalkersBody, data.top_talkers, renderTopTalkerRow);
        updateTable(elements.blockedIpsBody, data.blocked_ips, renderBlockedIpRow);
        
        updateEvents(data.events);
        if (trafficChart) {
            updateChart(total60);
        }
    }

    function updateTable(tbody, items, rowRenderer) {
        tbody.innerHTML = '';
        if (!items || items.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 6;
            cell.textContent = 'No data available.';
            cell.style.textAlign = 'center';
            cell.style.color = '#8b949e';
        } else {
            items.forEach(item => tbody.appendChild(rowRenderer(item)));
        }
    }
    
    function renderTopTalkerRow(item) {
        const row = document.createElement('tr');
        const probaClass = item.proba > 0.8 ? 'proba-high' : item.proba > 0.5 ? 'proba-med' : 'proba-low';

        row.innerHTML = `
            <td>${item.ip}</td>
            <td>${item.ip_rate_10s.toFixed(2)}</td>
            <td><span class="data-pill ${probaClass}">${item.proba.toFixed(2)}</span></td>
            <td><span class="data-pill share-pill">${item.share.toFixed(1)}%</span></td>
            <td>${item.paths}</td>
            <td class="actions"></td>`;

        const actionCell = row.querySelector('.actions');
        const blockButton = document.createElement('button');
        blockButton.textContent = 'Block';
        blockButton.className = 'btn-block';
        blockButton.disabled = item.is_blocked;
        if (!item.is_blocked) {
            blockButton.onclick = () => handleApiAction('block', item.ip);
        }
        actionCell.appendChild(blockButton);
        return row;
    }

    function renderBlockedIpRow(item) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.ip}</td>
            <td>${item.expires}</td>
            <td class="actions"></td>`;
        
        const actionCell = row.querySelector('.actions');
        const unblockButton = document.createElement('button');
        unblockButton.textContent = 'Unblock';
        unblockButton.className = 'btn-unblock';
        unblockButton.onclick = () => handleApiAction('unblock', item.ip);
        actionCell.appendChild(unblockButton);
        return row;
    }

    function updateEvents(events) {
        elements.eventsContainer.innerHTML = '';
        if (!events || events.length === 0) return;
        
        events.forEach(event => {
            const div = document.createElement('div');
            const timestamp = new Date(event.ts * 1000).toLocaleTimeString();
            const detailsSpan = event.details_text ? `<span class="event-details"> - ${event.details_text}</span>` : '';
            
            // Color-code events by attack type
            let kindClass = `kind-${event.kind}`;
            if (event.reason && event.reason.toLowerCase().includes('xss')) {
                kindClass += ' kind-xss';
            } else if (event.reason && event.reason.toLowerCase().includes('sql')) {
                kindClass += ' kind-sql';
            }
            
            div.className = `event-item ${kindClass}`;
            div.innerHTML = `[${timestamp}] <strong>${event.kind.toUpperCase()}</strong> - <span class="ip">${event.ip}</span> - ${event.reason}${detailsSpan}`;
            elements.eventsContainer.prepend(div); // Use prepend to show newest events at the top
        });
    }

    function updateChart(value) {
        trafficChart.data.labels.push(new Date().toLocaleTimeString());
        trafficChart.data.datasets[0].data.push(value);

        if (trafficChart.data.labels.length > CHART_MAX_POINTS) {
            trafficChart.data.labels.shift();
            trafficChart.data.datasets[0].data.shift();
        }
        trafficChart.update('none');
    }

    // --- API & WEBSOCKETS ---
    async function handleApiAction(action, ip) {
        try {
            const response = await fetch(`${API_BASE_URL}/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: ip })
            });
            if (!response.ok) {
                throw new Error(`Failed to ${action} IP`);
            }
        } catch (error) {
            console.error(`${action} failed:`, error);
        }
    }

    function connectWebSocket() {
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            elements.statusDot.className = 'status-dot status-connected';
            elements.statusText.textContent = 'Connected';
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateUI(data);
            } catch (e) {
                console.error("Failed to parse WebSocket message:", e);
            }
        };

        ws.onclose = () => {
            elements.statusDot.className = 'status-dot status-disconnected';
            elements.statusText.textContent = 'Disconnected. Retrying...';
            setTimeout(connectWebSocket, 3000); // Retry connection after 3 seconds
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            ws.close();
        };
    }

    // --- INITIALIZATION ---
    function init() {
        initChart();
        connectWebSocket();
    }

    init();
});
