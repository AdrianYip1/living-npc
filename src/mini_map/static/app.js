// UI shell only -- no backend wired up yet. The scrolling grid and player
// dot are a placeholder world so the camera-follow behavior is visible;
// WASD/arrow keys drive it directly for now and will be replaced once a
// real player position comes from the backend. Status values, play/pause,
// mic-hold, and the conversation log are likewise local-only.
//
// NPCs are drawn at random fixed positions purely to preview the visual --
// they don't move yet. Once game_agents can drive them, each NPC's
// position will come from its own LLM-controlled agent instead of
// spawnNPCs().
//
// Travelers are a second, separate category: instead of all spawning at
// once at start, one drifts in from a random map edge every so often,
// fades in, walks a straight placeholder line across the map, and is
// removed the moment it crosses back outside the map bounds.

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
const NPC_NAMES = ["Mira", "Talon", "Yuki", "Bram", "Sable"];
const NPC_COLORS = ["#e91e63", "#2196f3", "#ff9800", "#9c27b0", "#00bcd4"];
const TRAVELER_NAMES = ["Doran", "Kestrel", "Wren", "Fennic", "Ilsa", "Orin"];
const TRAVELER_SPEED = 70; // world units/sec
const TRAVELER_FADE_SECONDS = 1.2;
const TRAVELER_SPAWN_MIN_MS = 6000;
const TRAVELER_SPAWN_MAX_MS = 14000;
const MAX_TRAVELERS = 3;

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
// Only updated while a move key is held, so the arrow keeps pointing the
// last-faced direction once the player coasts to a stop.
let facingAngle = -Math.PI / 2; // start facing up

function spawnNPCs() {
  const count = 2 + Math.floor(Math.random() * 2); // 2 or 3
  const margin = 100;
  const names = [...NPC_NAMES].sort(() => Math.random() - 0.5);
  const npcs = [];
  for (let i = 0; i < count; i++) {
    npcs.push({
      name: names[i],
      color: NPC_COLORS[i % NPC_COLORS.length],
      x: (Math.random() * 2 - 1) * (MAP_HALF - margin),
      y: (Math.random() * 2 - 1) * (MAP_HALF - margin),
    });
  }
  return npcs;
}

const npcs = spawnNPCs();

// Traveler NPCs come and go over time, unlike the persistent npcs above.
// Mutated in place by spawnTraveler() (push) and tick() (splice on exit).
const travelers = [];

function spawnTraveler() {
  if (travelers.length >= MAX_TRAVELERS) {
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
    facingAngle = Math.atan2(dy, dx);
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
  proximityHint.classList.toggle("hidden", !nearbyNPC || state.conversationOpen);
}

function setPaused(paused) {
  state.paused = paused;
  statusPlaystate.textContent = paused ? "Paused" : "Playing";
  canvas.classList.toggle("paused", paused);
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

function openConversation(npc) {
  state.conversationOpen = true;
  conversationPartner = npc;
  conversationTarget.textContent = `Talking to ${npc.name}`;
  conversationEl.classList.remove("hidden");
  proximityHint.classList.add("hidden");
  conversationInput.focus();
}

function closeConversation() {
  state.conversationOpen = false;
  conversationPartner = null;
  conversationEl.classList.add("hidden");
  conversationInput.blur();
}

function isTypingTarget(target) {
  return target === conversationInput;
}

window.addEventListener("resize", resizeCanvas);

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

conversationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = conversationInput.value.trim();
  if (!text) {
    return;
  }
  appendLogLine(`> ${text}`);
  conversationInput.value = "";
  resizeConversationInput();
});

appendLogLine("Backend not connected yet -- input is only echoed locally.", { system: true });
resizeCanvas();
requestAnimationFrame(tick);
scheduleNextTraveler();
