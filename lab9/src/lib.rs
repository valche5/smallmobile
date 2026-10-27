//! Le laboratoire 9 hors jeu (`lab9/lab.md`) : les parties d'imitation de c130 découpées en plans
//! (`record`), le joueur qui écrit ses plans avec le décodeur (`agent`), et ce qu'il faut pour les
//! mesurer. L'arène, c130 et son moteur GPU viennent de lab7 (figé), tels quels.

pub mod agent;
pub mod fixture;
pub mod record;
pub mod selfplay;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use smallmobile_brain9::{encode, know, moves, net, plan, rand, search, tokens};
pub use smallmobile_lab7 as lab7;
