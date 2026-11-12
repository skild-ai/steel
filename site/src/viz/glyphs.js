// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Option glyphs for the reference: SteelVizGlyph(option) returns a small inline SVG string. */
window.SteelVizGlyph = (function () {
  /* option glyphs: one small icon per kind of knob */
  const GLYPH = {
    loop: '<path d="M14 6a6 6 0 1 0 1.5 5" /><path d="M14 2v4h-4"/>',
    cone: '<path d="M9 16 L3 4 M9 16 L15 4"/><ellipse cx="9" cy="4" rx="6" ry="2"/>',
    lift: '<path d="M2 15 H16"/><path d="M9 13 V4 M6 7 L9 4 L12 7"/>',
    joint: '<circle cx="9" cy="9" r="2.5"/><path d="M2 14 L7 10.5 M11 7.5 L16 4"/>',
    cap: '<rect x="2" y="6" width="14" height="6" rx="1"/><path d="M5 6v6M8 6v6M11 6v6"/>',
    tree: '<circle cx="9" cy="3.5" r="1.6"/><circle cx="4" cy="14" r="1.6"/><circle cx="14" cy="14" r="1.6"/><path d="M9 5v4M9 9L4 12.4M9 9l5 3.4"/>',
    moon: '<path d="M13 12.5A6 6 0 0 1 6.5 4a6 6 0 1 0 6.5 8.5z"/>',
    bolt: '<path d="M10 2 L4 10 H9 L8 16 L14 8 H9 Z"/>',
    clock: '<circle cx="9" cy="9" r="6.5"/><path d="M9 5v4l3 2"/>',
    equal: '<path d="M3 7h12M3 11h12"/>',
    bounce: '<path d="M2 15H16"/><path d="M3 4 Q6 14 9 15 Q11 9 13 15 Q14 12 15 15"/>',
    flag: '<path d="M4 16V3h9l-2 3 2 3H4"/>',
  };
  const GLYPH_OF = {
    "Solve": "loop", "Stabilization": "lift", "Contacts and friction": "cone", "Joints and drives": "joint",
    "Capacity and diagnostics": "cap", "Propagation response": "tree", "Sleeping": "moon", "Performance": "bolt",
  };
  const GLYPH_OVERRIDE = {
    update_mass_matrix_interval: "clock", deterministic: "equal", enable_restitution: "bounce", restitution_velocity_threshold: "bounce",
    restitution_combine: "bounce", row_watermark: "flag", warn_constraint_overflow: "flag",
  };
  return (o) => `<svg class="oi" viewBox="0 0 18 18" aria-hidden="true">${GLYPH[GLYPH_OVERRIDE[o.name] || GLYPH_OF[o.category] || "loop"]}</svg>`;
})();
