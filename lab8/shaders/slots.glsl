// Les places de la mémoire du GPU (`vulkan.rs`, décodeur) : une lecture de position par place (sortie
// de l'encodeur des régions, `dec_reg` des régions, clés et valeurs de l'attention croisée de chaque
// couche du décodeur, en bf16 ; jetons présents et nombre de régions), et le cache de chaque voie du
// décodeur (clés et valeurs de l'attention causale, par couche et par pas, en f32).
layout(std430, binding = 6) buffer Slots { bfloat16_t slot16[]; };
layout(std430, binding = 7) buffer SlotInfo { uint slot_info[]; }; // par place : régions, puis 60 drapeaux
layout(std430, binding = 8) buffer Caches { float cache[]; };
layout(std430, binding = 9) readonly buffer Lanes { uint lanes[]; }; // par voie : LANE_WORDS mots
layout(std430, binding = 10) writeonly buffer Logits { float logits[]; };

const uint INFO = 64u; // mots par place dans slot_info
// une voie : place, cache, pas (0 au premier), coup précédent (65535 : aucun), nombre de coups déjà
// joués sur la région du coup précédent, puis les 20 entrées du pas (une par mot)
const uint LANE_WORDS = 32u;
const uint MAX_STEPS = 128u; // pas au plus par plan (cache)
const uint L_SLOT = 0u, L_CACHE = 1u, L_STEP = 2u, L_PREV = 3u, L_COUNT = 4u, L_DYN = 8u;
