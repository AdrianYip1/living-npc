// The scrolling grid and player dot are still a local placeholder -- WASD/
// arrow keys drive the player directly, and will be replaced once a real
// player position comes from the backend. Mic-hold and the conversation
// log's typing/rendering are likewise still local-only.
//
// NPCs, though, are real: their positions come from polling /api/state on
// the mini_map server, which runs game_agents' NPCRegistry behind an
// autonomous time-of-day tick loop (see mini_map/simulation.py). Backend
// coordinates run -100..100 on each axis (see game_agents/world.py);
// WORLD_SCALE below maps that onto this canvas's own (larger, arbitrary)
// world-unit space.
//
// Play/pause is real too: it POSTs to /api/pause and /api/resume, which
// actually stop/resume the server's world tick loop (environment clock +
// NPC LLM queries), not just a client-side dim. state.paused also gates
// player movement and traveler spawning locally, and gets resynced from
// the server's own `paused` flag on every /api/state poll.
//
// Travelers are still a separate, client-only placeholder category: one
// drifts in from a random map edge every so often, fades in, walks a
// straight placeholder line across the map, and is removed the moment it
// crosses back outside the map bounds. Spawning real travelers from the
// environment agent is a natural next step, not done yet.

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const GRID_SIZE = 48;
const NPC_RADIUS = 16;
const PLAYER_RADIUS = NPC_RADIUS;
const INTERACT_RANGE = 90; // world units; how close the player must be to talk
const MAP_SIZE = 1200; // world units, arbitrary for now
const MAP_HALF = MAP_SIZE / 2;
const MAX_SPEED = 260; // world px/sec
const ACCEL = 900; // px/sec^2 while a move key is held
const DECEL = 1400; // px/sec^2 once keys are released
const NPC_COLORS = ["#e91e63", "#2196f3", "#ff9800", "#9c27b0", "#00bcd4"];
const TRAVELER_NAMES = ["Doran", "Kestrel", "Wren", "Fennic", "Ilsa", "Orin"];
const TRAVELER_SPEED = 70; // world units/sec
const TRAVELER_FADE_SECONDS = 1.2;
const TRAVELER_SPAWN_MIN_MS = 6000;
const TRAVELER_SPAWN_MAX_MS = 14000;
const MAX_TRAVELERS = 3;
const GAME_BOUND = 100; // matches game_agents/world.py's MAP_MIN/MAX
const WORLD_SCALE = MAP_HALF / GAME_BOUND; // backend coord -> canvas world unit
const STATE_POLL_MS = 2000;

const statusTime = document.getElementById("status-time");
const statusWeather = document.getElementById("status-weather");
const statusTickInterval = document.getElementById("status-tick-interval");
const statusNpcQueryInterval = document.getElementById("status-npc-query-interval");
const statusQueriesPerMinute = document.getElementById("status-queries-per-minute");
const statusPlaystate = document.getElementById("status-playstate");
const statusMic = document.getElementById("status-mic");
const conversationEl = document.getElementById("conversation");
const conversationTarget = document.getElementById("conversation-target");
const conversationInput = document.getElementById("conversation-input");
const conversationForm = document.getElementById("conversation-form");
const conversationLog = document.getElementById("conversation-log");
const proximityHint = document.getElementById("proximity-hint");

const state = {
  paused: true,
  micActive: false,
  conversationOpen: false,
};

// The NPC in interaction range this frame, if any; updated every tick().
let nearbyNPC = null;

// World position of the player. The player is always drawn at the screen
// center; the grid scrolls under them by this offset instead. Clamped to
// [-MAP_HALF, MAP_HALF] so the player can see past the map edge but never
// cross it.
const camera = { x: 0, y: 0 };
const velocity = { x: 0, y: 0 };

// Direction the player is facing, in radians (0 = right, screen-space).
// Derived from the actual velocity vector each frame (not the raw 8-way
// key input) so the arrow follows the curve of accel/decel through turns
// instead of snapping between 8 fixed angles. Held steady below
// FACING_MIN_SPEED so it doesn't jitter to noise while coasting to a stop.
let facingAngle = -Math.PI / 2; // start facing up
const FACING_MIN_SPEED = 4; // world units/sec

// Real NPCs, kept live by polling /api/state. Colors are assigned once per
// name (on first sighting) and then held stable across polls, since the
// backend doesn't send one -- npcColors persists that assignment even
// though the npc objects themselves get replaced each poll.
const npcs = [];
const npcColors = new Map();

function colorFor(name) {
  if (!npcColors.has(name)) {
    npcColors.set(name, NPC_COLORS[npcColors.size % NPC_COLORS.length]);
  }
  return npcColors.get(name);
}

function applyState(data) {
  if (typeof data.clock === "string") {
    statusTime.textContent = data.clock;
  }
  if (typeof data.weather === "string") {
    statusWeather.textContent = data.weather;
  }
  // These two are launch-time constants (--tick-interval /
  // --npc-query-interval), but reading them off the server's own state
  // rather than hardcoding them keeps this display honest if the backend
  // was started with non-default values.
  if (typeof data.tick_interval_s === "number") {
    statusTickInterval.textContent = `${data.tick_interval_s}s`;
  }
  if (typeof data.npc_query_interval_s === "number") {
    statusNpcQueryInterval.textContent = `${data.npc_query_interval_s}s`;
  }
  if (typeof data.queries_per_game_minute === "number") {
    statusQueriesPerMinute.textContent = data.queries_per_game_minute.toFixed(2);
  }
  if (typeof data.paused === "boolean") {
    // Server-authoritative sync, not a user action -- updates the label/
    // dimming to match reality (e.g. after a reload) without re-POSTing.
    applyPausedUI(data.paused);
  }

  const byName = new Map(npcs.map((npc) => [npc.name, npc]));
  npcs.length = 0;
  for (const entry of data.npcs || []) {
    const existing = byName.get(entry.name);
    const npc = existing || { name: entry.name, color: colorFor(entry.name) };
    npc.x = entry.x * WORLD_SCALE;
    npc.y = entry.y * WORLD_SCALE;
    npc.busy = entry.busy;
    npc.activity = entry.activity;
    npcs.push(npc);
  }

  if (nearbyNPC && !npcs.includes(nearbyNPC)) {
    nearbyNPC = null;
  }
}

let statePollFailed = false;

async function pollState() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    applyState(await response.json());
    statePollFailed = false;
  } catch (err) {
    if (!statePollFailed) {
      appendLogLine(`Lost connection to the backend (${err.message}). Retrying...`, { system: true });
      statePollFailed = true;
    }
  }
}

// Traveler NPCs come and go over time, unlike the persistent npcs above.
// Mutated in place by spawnTraveler() (push) and tick() (splice on exit).
const travelers = [];

function spawnTraveler() {
  if (state.paused || travelers.length >= MAX_TRAVELERS) {
    return;
  }

  const edge = Math.floor(Math.random() * 4);
  const along = (Math.random() * 2 - 1) * (MAP_HALF - 20);
  const inset = MAP_HALF - 4;
  let x;
  let y;
  let inwardAngle;
  if (edge === 0) {
    x = along;
    y = -inset;
    inwardAngle = Math.PI / 2; // spawned on top edge, walks down
  } else if (edge === 1) {
    x = inset;
    y = along;
    inwardAngle = Math.PI; // spawned on right edge, walks left
  } else if (edge === 2) {
    x = along;
    y = inset;
    inwardAngle = -Math.PI / 2; // spawned on bottom edge, walks up
  } else {
    x = -inset;
    y = along;
    inwardAngle = 0; // spawned on left edge, walks right
  }

  // Aim roughly across the map, not straight along the edge, with some
  // spread so travelers don't all cut the exact same line.
  const angle = inwardAngle + (Math.random() * (Math.PI / 3) - Math.PI / 6);
  const speed = TRAVELER_SPEED * (0.8 + Math.random() * 0.4);

  const taken = new Set(npcs.concat(travelers).map((entity) => entity.name));
  const available = TRAVELER_NAMES.filter((name) => !taken.has(name));
  const pool = available.length > 0 ? available : TRAVELER_NAMES;

  travelers.push({
    name: pool[Math.floor(Math.random() * pool.length)],
    color: NPC_COLORS[Math.floor(Math.random() * NPC_COLORS.length)],
    isTraveler: true, // no backend counterpart -- conversation panel stays local-only for these
    x,
    y,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    opacity: 0,
  });
}

function scheduleNextTraveler() {
  const delay = TRAVELER_SPAWN_MIN_MS + Math.random() * (TRAVELER_SPAWN_MAX_MS - TRAVELER_SPAWN_MIN_MS);
  setTimeout(() => {
    spawnTraveler();
    scheduleNextTraveler();
  }, delay);
}

const pressedKeys = new Set();
const MOVE_KEYS = {
  KeyW: [0, -1],
  KeyS: [0, 1],
  KeyA: [-1, 0],
  KeyD: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
};

// Logical (CSS-pixel) canvas size. The backing store is sized up by
// devicePixelRatio in resizeCanvas() so circles and text render sharp on
// HiDPI displays instead of being upscaled and blurry; every draw call
// below works in these logical units, not canvas.width/height directly.
let viewWidth = window.innerWidth;
let viewHeight = window.innerHeight;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  viewWidth = window.innerWidth;
  viewHeight = window.innerHeight;
  canvas.width = Math.round(viewWidth * dpr);
  canvas.height = Math.round(viewHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Shared circle-with-initial rendering for both persistent NPCs and
// travelers. `dashed` gives travelers a visibly different ring so they
// read as a different kind of NPC; `opacity` drives their fade in/out.
function drawNpcIcon(entity, sx, sy, { nearby = false, opacity = 1, dashed = false } = {}) {
  ctx.save();
  ctx.globalAlpha = opacity;

  if (nearby) {
    ctx.beginPath();
    ctx.arc(sx, sy, NPC_RADIUS + 5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(sx, sy, NPC_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = entity.color;
  ctx.fill();
  if (dashed) {
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const initial = entity.name[0].toUpperCase();
  ctx.font = `bold ${Math.round(NPC_RADIUS * 1.1)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  // "middle" baseline centers on font-wide metrics, which sits a capital
  // letter a few px off true center; measure this glyph's actual box and
  // center on that instead.
  const metrics = ctx.measureText(initial);
  const letterY = sy + (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.strokeText(initial, sx, letterY);
  ctx.fillStyle = "#fff";
  ctx.fillText(initial, sx, letterY);

  ctx.restore();
}

function drawWorld() {
  const width = viewWidth;
  const height = viewHeight;

  // Void beyond the map border -- visible once the player nears an edge.
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, width, height);

  const mapLeft = width / 2 + (-MAP_HALF - camera.x);
  const mapTop = height / 2 + (-MAP_HALF - camera.y);

  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(mapLeft, mapTop, MAP_SIZE, MAP_SIZE);

  ctx.save();
  ctx.beginPath();
  ctx.rect(mapLeft, mapTop, MAP_SIZE, MAP_SIZE);
  ctx.clip();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  const offsetX = ((camera.x % GRID_SIZE) + GRID_SIZE) % GRID_SIZE;
  const offsetY = ((camera.y % GRID_SIZE) + GRID_SIZE) % GRID_SIZE;

  for (let x = -offsetX; x <= width; x += GRID_SIZE) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = -offsetY; y <= height; y += GRID_SIZE) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  for (const npc of npcs) {
    const sx = width / 2 + (npc.x - camera.x);
    const sy = height / 2 + (npc.y - camera.y);
    drawNpcIcon(npc, sx, sy, { nearby: npc === nearbyNPC });
  }

  for (const traveler of travelers) {
    const sx = width / 2 + (traveler.x - camera.x);
    const sy = height / 2 + (traveler.y - camera.y);
    drawNpcIcon(traveler, sx, sy, { nearby: traveler === nearbyNPC, opacity: traveler.opacity, dashed: true });
  }

  ctx.restore();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 2;
  ctx.strokeRect(mapLeft, mapTop, MAP_SIZE, MAP_SIZE);

  ctx.beginPath();
  ctx.arc(width / 2, height / 2, PLAYER_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = "#4caf50";
  ctx.fill();

  ctx.save();
  ctx.translate(width / 2, height / 2);
  ctx.rotate(facingAngle);
  ctx.beginPath();
  ctx.moveTo(PLAYER_RADIUS * 0.55, 0);
  ctx.lineTo(-PLAYER_RADIUS * 0.35, PLAYER_RADIUS * 0.45);
  ctx.lineTo(-PLAYER_RADIUS * 0.35, -PLAYER_RADIUS * 0.45);
  ctx.closePath();
  ctx.fillStyle = "rgba(10, 30, 15, 0.85)";
  ctx.fill();
  ctx.restore();
}

let lastFrameTime = performance.now();

function tick(now) {
  const dt = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  if (!state.paused) {
    let dx = 0;
    let dy = 0;
    for (const key of pressedKeys) {
      const dir = MOVE_KEYS[key];
      if (dir) {
        dx += dir[0];
        dy += dir[1];
      }
    }

    if (dx !== 0 || dy !== 0) {
      const length = Math.hypot(dx, dy);
      velocity.x += (dx / length) * ACCEL * dt;
      velocity.y += (dy / length) * ACCEL * dt;
      const speed = Math.hypot(velocity.x, velocity.y);
      if (speed > MAX_SPEED) {
        velocity.x = (velocity.x / speed) * MAX_SPEED;
        velocity.y = (velocity.y / speed) * MAX_SPEED;
      }
    } else {
      const speed = Math.hypot(velocity.x, velocity.y);
      if (speed > 0) {
        const nextSpeed = Math.max(0, speed - DECEL * dt);
        const scale = nextSpeed / speed;
        velocity.x *= scale;
        velocity.y *= scale;
      }
    }

    const currentSpeed = Math.hypot(velocity.x, velocity.y);
    if (currentSpeed > FACING_MIN_SPEED) {
      facingAngle = Math.atan2(velocity.y, velocity.x);
    }

    camera.x += velocity.x * dt;
    camera.y += velocity.y * dt;

    if (camera.x > MAP_HALF) {
      camera.x = MAP_HALF;
      velocity.x = 0;
    } else if (camera.x < -MAP_HALF) {
      camera.x = -MAP_HALF;
      velocity.x = 0;
    }
    if (camera.y > MAP_HALF) {
      camera.y = MAP_HALF;
      velocity.y = 0;
    } else if (camera.y < -MAP_HALF) {
      camera.y = -MAP_HALF;
      velocity.y = 0;
    }

    updateTravelers(dt);
  }

  updateNearbyNPC();
  drawWorld();
  requestAnimationFrame(tick);
}

function updateTravelers(dt) {
  for (let i = travelers.length - 1; i >= 0; i--) {
    const traveler = travelers[i];
    traveler.x += traveler.vx * dt;
    traveler.y += traveler.vy * dt;
    traveler.opacity = Math.min(1, traveler.opacity + dt / TRAVELER_FADE_SECONDS);

    if (Math.abs(traveler.x) > MAP_HALF || Math.abs(traveler.y) > MAP_HALF) {
      travelers.splice(i, 1);
      if (nearbyNPC === traveler) {
        nearbyNPC = null;
      }
      if (conversationPartner === traveler) {
        closeConversation();
      }
    }
  }
}

function updateNearbyNPC() {
  let closest = null;
  let closestDist = INTERACT_RANGE;
  for (const entity of npcs.concat(travelers)) {
    const dist = Math.hypot(entity.x - camera.x, entity.y - camera.y);
    if (dist < closestDist) {
      closest = entity;
      closestDist = dist;
    }
  }
  nearbyNPC = closest;
  proximityHint.classList.toggle("hidden", !nearbyNPC || nearbyNPC.busy || state.conversationOpen);
}

function applyPausedUI(paused) {
  state.paused = paused;
  statusPlaystate.textContent = paused ? "Paused" : "Playing";
  canvas.classList.toggle("paused", paused);
}

// The player-facing action: updates locally right away (feels instant) and
// tells the server to actually stop/resume the world tick loop -- while
// paused, the environment clock stops and NPCs stop getting queried, not
// just a cosmetic dim (see mini_map/simulation.py's Simulation.pause()).
async function setPaused(paused) {
  applyPausedUI(paused);
  try {
    await postJSON(paused ? "/api/pause" : "/api/resume", {});
  } catch (err) {
    // best effort -- the next state poll resyncs from server truth anyway
  }
}

function setMicActive(active) {
  state.micActive = active;
  statusMic.textContent = active ? "Listening..." : "Off";
  statusMic.classList.toggle("active", active);
}

function appendLogLine(text, { system = false } = {}) {
  const line = document.createElement("div");
  line.className = system ? "line system" : "line";
  line.textContent = text;
  conversationLog.appendChild(line);
  conversationLog.scrollTop = conversationLog.scrollHeight;
}

let conversationPartner = null;

async function postJSON(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// Real NPCs (not travelers -- they have no backend counterpart to claim
// busy state on) go through the server to actually start the exchange:
// this is the same busy claim the world tick and NPC-to-NPC conversations
// use, so a real conversation locks this NPC out of autonomous behavior
// for as long as the panel stays open, exactly like being mid-exchange
// with another NPC would.
async function openConversation(npc) {
  state.conversationOpen = true;
  conversationPartner = npc;
  conversationTarget.textContent = `Talking to ${npc.name}`;
  conversationEl.classList.remove("hidden");
  proximityHint.classList.add("hidden");
  conversationLog.innerHTML = "";
  conversationInput.focus();

  if (!npc.isTraveler) {
    let ok = false;
    try {
      const response = await postJSON("/api/conversation/start", { name: npc.name });
      ok = response.ok;
    } catch (err) {
      ok = false;
    }
    if (!ok) {
      appendLogLine(`${npc.name} is busy right now.`, { system: true });
      closeConversation();
    }
  }
}

function closeConversation() {
  state.conversationOpen = false;
  const npc = conversationPartner;
  conversationPartner = null;
  conversationEl.classList.add("hidden");
  conversationInput.blur();
  if (npc && !npc.isTraveler) {
    postJSON("/api/conversation/end", { name: npc.name }).catch(() => {});
  }
}

function setConversationBusy(busy) {
  conversationInput.disabled = busy;
  conversationForm.querySelector("button").disabled = busy;
}

function describeAction(action) {
  if (action.name === "move_to") return "walks off";
  if (action.name === "wait") return "doesn't answer";
  if (action.name === "initiate_conversation") return "turns to talk to someone else";
  return action.name;
}

function isTypingTarget(target) {
  return target === conversationInput;
}

window.addEventListener("resize", resizeCanvas);

statusPlaystate.addEventListener("click", () => setPaused(!state.paused));

document.addEventListener("keydown", (event) => {
  if (event.code === "Escape" && state.conversationOpen) {
    event.preventDefault();
    closeConversation();
    return;
  }

  if (isTypingTarget(event.target)) {
    return;
  }

  if (event.code === "Space") {
    event.preventDefault();
    setPaused(!state.paused);
    return;
  }

  if (event.code === "Tab" && !state.micActive) {
    event.preventDefault();
    setMicActive(true);
    return;
  }

  if (event.code === "KeyE") {
    event.preventDefault();
    if (state.conversationOpen) {
      closeConversation();
    } else if (nearbyNPC) {
      openConversation(nearbyNPC);
    }
    return;
  }

  if (event.code in MOVE_KEYS) {
    event.preventDefault();
    pressedKeys.add(event.code);
  }
});

document.addEventListener("keyup", (event) => {
  if (event.code === "Tab") {
    setMicActive(false);
  }
  pressedKeys.delete(event.code);
});

window.addEventListener("blur", () => {
  pressedKeys.clear();
  setMicActive(false);
});

function resizeConversationInput() {
  conversationInput.style.height = "auto";
  conversationInput.style.height = `${conversationInput.scrollHeight}px`;
}

conversationInput.addEventListener("input", resizeConversationInput);

conversationInput.addEventListener("keydown", (event) => {
  if (event.code === "Enter" && !event.shiftKey) {
    event.preventDefault();
    conversationForm.requestSubmit();
  }
});

conversationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = conversationInput.value.trim();
  const npc = conversationPartner;
  if (!text || !npc) {
    return;
  }
  appendLogLine(`> ${text}`);
  conversationInput.value = "";
  resizeConversationInput();

  if (npc.isTraveler) {
    return; // no backend counterpart -- stays a local echo, as before
  }

  setConversationBusy(true);
  try {
    const response = await postJSON("/api/conversation/say", { name: npc.name, text });
    if (!response.ok) {
      appendLogLine(`${npc.name} didn't respond.`, { system: true });
      return;
    }
    const data = await response.json();
    if (data.utterance) {
      appendLogLine(`${npc.name}: ${data.utterance}`);
    } else if (data.action) {
      appendLogLine(`* ${npc.name} ${describeAction(data.action)} *`, { system: true });
    }
  } catch (err) {
    appendLogLine(`Lost connection while talking to ${npc.name}.`, { system: true });
  } finally {
    setConversationBusy(false);
    conversationInput.focus();
  }
});

appendLogLine("Connecting to the living-npc backend...", { system: true });
resizeCanvas();
requestAnimationFrame(tick);
scheduleNextTraveler();
pollState();
setInterval(pollState, STATE_POLL_MS);
