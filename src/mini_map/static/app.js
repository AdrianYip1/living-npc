// The scrolling grid and player dot are still a local placeholder -- WASD/
// arrow keys drive the player directly, and will be replaced once a real
// player position comes from the backend. The conversation log's
// typing/rendering is likewise still local-only.
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
// player movement locally, and gets resynced from the server's own
// `paused` flag on every /api/state poll.
//
// Travelers are decided by the environment agent (how many per in-game
// day, and when -- see environment_agent/agent.py) and then run by the
// server as live LLM agents, like the persistent NPCs: they arrive in
// /api/state's npcs list flagged `traveler`, walk wherever their own
// move_to calls send them, and vanish from the list once they reach their
// exit point. The page only draws them (dashed ring, fade-in) and logs
// each arrival from traveler_arrivals.

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
const PERSISTENT_NPC_COLOR = "#2196f3";
const TRAVELER_COLOR = "#ff9800";
const TRAVELER_FADE_SECONDS = 1.2;
const GAME_BOUND = 100; // matches game_agents/world.py's MAP_MIN/MAX
const WORLD_SCALE = MAP_HALF / GAME_BOUND; // backend coord -> canvas world unit
const STATE_POLL_MS = 200;
// Client-side smoothing of server-driven NPC positions (see applyState).
const NPC_SNAP_DISTANCE = 60; // world units; bigger gaps than this just jump
const NPC_CORRECTION_RATE = 8; // per sec; how fast drift from the server is bled off

const statusTime = document.getElementById("status-time");
const statusWeather = document.getElementById("status-weather");
const statusGameMinutesPerRealMinute = document.getElementById("status-game-minutes-per-real-minute");
const statusLlmCallsPerGameHour = document.getElementById("status-llm-calls-per-game-hour");
const statusLlmCallsPerRealMinute = document.getElementById("status-llm-calls-per-real-minute");
const statusPlaystate = document.getElementById("status-playstate");
const conversationEl = document.getElementById("conversation");
const conversationTarget = document.getElementById("conversation-target");
const conversationInput = document.getElementById("conversation-input");
const conversationForm = document.getElementById("conversation-form");
const conversationLog = document.getElementById("conversation-log");
const proximityHint = document.getElementById("proximity-hint");

const state = {
  paused: true,
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

// Real NPCs, kept live by polling /api/state. All share PERSISTENT_NPC_COLOR
// so they read apart from travelers at a glance.
const npcs = [];

function applyState(data) {
  if (typeof data.clock === "string") {
    statusTime.textContent = data.clock;
  }
  if (typeof data.weather === "string") {
    statusWeather.textContent = data.weather;
  }
  // The first (game_minutes_per_real_minute, derived server-side from
  // --ticks-per-real-minute and EnvironmentAgent's minutes_per_tick) and
  // second (llm_calls_per_game_hour, a launch-time constant) come off the
  // server's own state rather than being hardcoded, so this stays honest
  // if the backend was started with non-default values. The third is
  // derived server-side from the other two (see
  // Simulation._llm_calls_per_real_minute).
  if (typeof data.game_minutes_per_real_minute === "number") {
    statusGameMinutesPerRealMinute.textContent = data.game_minutes_per_real_minute.toFixed(2).replace(/\.00$/, "");
  }
  if (typeof data.llm_calls_per_game_hour === "number") {
    statusLlmCallsPerGameHour.textContent = data.llm_calls_per_game_hour.toFixed(2).replace(/\.00$/, "");
  }
  if (typeof data.llm_calls_per_real_minute === "number") {
    statusLlmCallsPerRealMinute.textContent = data.llm_calls_per_real_minute.toFixed(2).replace(/\.00$/, "");
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
    const npc = existing || {
      name: entry.name,
      isTraveler: Boolean(entry.traveler),
      color: entry.traveler ? TRAVELER_COLOR : PERSISTENT_NPC_COLOR,
      // Travelers fade in on first sighting; residents are just there.
      opacity: entry.traveler ? 0 : 1,
      errX: 0,
      errY: 0,
    };
    const serverX = entry.x * WORLD_SCALE;
    const serverY = entry.y * WORLD_SCALE;
    // The server owns where NPCs actually are (see NPCRegistry.
    // step_movement); updateNpcs() re-runs the same walking physics every
    // frame so motion stays smooth between polls, and any drift from the
    // server's position gets bled off over the next few frames instead of
    // snapping -- unless it's too big to be drift (first sighting, etc.).
    if (existing && Math.hypot(serverX - npc.x, serverY - npc.y) < NPC_SNAP_DISTANCE) {
      npc.errX = serverX - npc.x;
      npc.errY = serverY - npc.y;
    } else {
      npc.x = serverX;
      npc.y = serverY;
      npc.errX = 0;
      npc.errY = 0;
    }
    npc.vx = (entry.vx || 0) * WORLD_SCALE;
    npc.vy = (entry.vy || 0) * WORLD_SCALE;
    npc.destination = Array.isArray(entry.destination)
      ? { x: entry.destination[0] * WORLD_SCALE, y: entry.destination[1] * WORLD_SCALE }
      : null;
    npc.busy = entry.busy;
    npc.activity = entry.activity;
    npcs.push(npc);
  }

  if (nearbyNPC && !npcs.includes(nearbyNPC)) {
    nearbyNPC = null;
  }
  // The server won't let anyone leave mid-conversation, but if the partner
  // is gone anyway (e.g. a server restart), don't leave the panel talking
  // to no one.
  if (conversationPartner && !npcs.includes(conversationPartner)) {
    closeConversation();
  }

  logTravelerArrivals(data.traveler_arrivals || []);
}

// Highest traveler_arrivals id already logged. null until the first poll,
// which only records where the list stands -- otherwise a page reload
// would re-log every recent arrival at once.
let lastTravelerId = null;

function logTravelerArrivals(arrivals) {
  const maxId = arrivals.reduce((max, arrival) => Math.max(max, arrival.id), 0);
  if (lastTravelerId === null) {
    lastTravelerId = maxId;
    return;
  }
  for (const arrival of arrivals) {
    if (arrival.id > lastTravelerId) {
      appendLogLine(`A traveler arrives at ${arrival.arrived_at}: ${arrival.identity.name}. ${arrival.identity.backstory}`, { system: true });
    }
  }
  lastTravelerId = Math.max(lastTravelerId, maxId);
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
    drawNpcIcon(npc, sx, sy, { nearby: npc === nearbyNPC, opacity: npc.opacity, dashed: npc.isTraveler });
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

    updateNpcs(dt);
  }

  updateNearbyNPC();
  drawWorld();
  requestAnimationFrame(tick);
}

// JS twin of game_agents/world.py's step_toward(), in canvas units: walk
// toward `destination` (or brake to a stop if null) with the player's own
// MAX_SPEED / ACCEL / DECEL, braking early enough to stop on the spot.
function stepToward(body, destination, dt) {
  const speed = Math.hypot(body.vx, body.vy);

  if (!destination) {
    if (speed > 0) {
      const scale = Math.max(0, speed - DECEL * dt) / speed;
      body.vx *= scale;
      body.vy *= scale;
    }
  } else {
    const toX = destination.x - body.x;
    const toY = destination.y - body.y;
    const dist = Math.hypot(toX, toY);
    if (dist <= Math.max(speed * dt, 1e-6)) {
      body.x = destination.x;
      body.y = destination.y;
      body.vx = 0;
      body.vy = 0;
      return;
    }

    const desiredSpeed = Math.min(MAX_SPEED, Math.sqrt(2 * DECEL * dist));
    let dvx = (toX / dist) * desiredSpeed - body.vx;
    let dvy = (toY / dist) * desiredSpeed - body.vy;
    const dv = Math.hypot(dvx, dvy);
    const maxDv = (desiredSpeed < speed ? DECEL : ACCEL) * dt;
    if (dv > maxDv) {
      dvx = (dvx / dv) * maxDv;
      dvy = (dvy / dv) * maxDv;
    }
    body.vx += dvx;
    body.vy += dvy;
  }

  body.x += body.vx * dt;
  body.y += body.vy * dt;
}

function updateNpcs(dt) {
  const k = Math.min(1, NPC_CORRECTION_RATE * dt);
  for (const npc of npcs) {
    stepToward(npc, npc.busy ? null : npc.destination, dt);
    npc.x += npc.errX * k;
    npc.y += npc.errY * k;
    npc.errX -= npc.errX * k;
    npc.errY -= npc.errY * k;
    npc.opacity = Math.min(1, npc.opacity + dt / TRAVELER_FADE_SECONDS);
  }
}

function updateNearbyNPC() {
  let closest = null;
  let closestDist = INTERACT_RANGE;
  for (const entity of npcs) {
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

// Every NPC, resident or traveler, goes through the server to actually
// start the exchange: this is the same busy claim the world tick and NPC-to-NPC conversations
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

function closeConversation() {
  state.conversationOpen = false;
  const npc = conversationPartner;
  conversationPartner = null;
  conversationEl.classList.add("hidden");
  conversationInput.blur();
  if (npc) {
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
  pressedKeys.delete(event.code);
});

window.addEventListener("blur", () => {
  pressedKeys.clear();
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
pollState();
setInterval(pollState, STATE_POLL_MS);
