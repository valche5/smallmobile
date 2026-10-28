// Commun aux shaders de lab9 (`lab9/src/vulkan.rs`) : les tampons, une position = ses octets
// d'entrée (`POS_BYTES` : régions 48 × 77, partie 820, distances 48 × 48, nombre de régions et de
// joueurs, place ; puis la première ligne de la position parmi les lignes des seuls jetons présents).
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_bfloat16 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(std430, binding = 0) readonly buffer Weights { float w[]; };
layout(std430, binding = 1) readonly buffer Weights16 { bfloat16_t wb[]; };
layout(std430, binding = 2) buffer Act { float act[]; };
layout(std430, binding = 3) readonly buffer Input { uint inp[]; };
layout(std430, binding = 4) writeonly buffer Out16 { bfloat16_t out16[]; };
layout(std430, binding = 5) writeonly buffer OutF { float outf[]; };
// les activations lues par les produits, en bf16 (sorties des normalisations, de l'attention, de gelu)
layout(std430, binding = 11) buffer Act16 { bfloat16_t act16[]; };

const uint R = 48u;
const uint NODE = 77u;
const uint GLOBAL = 820u;
const uint TOKENS = 60u;
const uint T_PLAYER = 48u;
const uint T_MARKET = 53u;
const uint T_GAME = 59u;
const uint IN_NODE = 0u;
const uint IN_GLOB = R * NODE;
const uint IN_DIST = IN_GLOB + GLOBAL;
const uint IN_REGIONS = IN_DIST + R * R;
const uint IN_ROWS = IN_REGIONS + 4u;
const uint POS_BYTES = IN_ROWS + 4u;
const uint G_PLAYER = 67u;
const uint P = 81u;
const uint G_MARKET = 472u;
const uint M = 38u;

uint byte_at(uint pos, uint at) {
    uint a = pos * POS_BYTES + at;
    return (inp[a >> 2] >> ((a & 3u) * 8u)) & 0xffu;
}

uint regions_of(uint pos) { return byte_at(pos, IN_REGIONS); }

// la première ligne de la position parmi les lignes des jetons présents (les blocs de l'encodeur ne
// calculent que celles-là ; `gather.comp`)
uint row_start(uint pos) { return inp[(pos * POS_BYTES + IN_ROWS) >> 2]; }

// le jeton `t` de la position est-il présent ?
bool token_valid(uint pos, uint t) {
    if (t < R) return t < regions_of(pos);
    if (t < T_MARKET) return byte_at(pos, IN_GLOB + G_PLAYER + (t - T_PLAYER) * P) != 0u;
    if (t < T_GAME) return byte_at(pos, IN_GLOB + G_MARKET + (t - T_MARKET) * M) != 0u;
    return true;
}
