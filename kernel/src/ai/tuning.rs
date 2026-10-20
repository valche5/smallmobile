//! Réglages de l'IA. Les interrupteurs de modèles valent 0 (coupé) ou 1.

macro_rules! tuning {
    ($($(#[$doc:meta])* $field:ident = $default:expr),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Tuning { $($(#[$doc])* pub $field: f64),+ }

        impl Default for Tuning {
            fn default() -> Self {
                Tuning { $($field: $default),+ }
            }
        }

        impl Tuning {
            /// Change un réglage d'après son nom ; faux si le nom est inconnu.
            pub fn set(&mut self, name: &str, value: f64) -> bool {
                match name {
                    $(stringify!($field) => self.$field = value,)+
                    _ => return false,
                }
                true
            }
        }
    };
}

tuning! {
    /// Bonus par région voisine déjà tenue par le peuple : garder un territoire d'un seul tenant.
    adjacency = 0.12,
    /// Avance, en pièces et en part du total, à partir de laquelle un joueur « domine clairement ».
    dom_min = 10.0,
    dom_share = 0.2,
    /// Bonus pour reprendre une région au joueur dominant : de `dom_base` à `dom_max` selon son avance.
    dom_base = 0.6,
    dom_max = 1.2,
    /// Exposant du coût dans le rendement d'une conquête (1 : pièces par jeton engagé).
    cost_power = 1.0,
    /// Rendement en dessous duquel l'IA garde ses jetons pour la défense (0 : conquérir tant que possible).
    min_ratio = 0.0,
    /// Bonus pour prendre une région à un peuple actif / en déclin adverse, même s'il ne domine pas.
    foe = 0.0,
    foe_declined = 0.0,
    /// Difficile : bonus pour toute région prise à un adversaire, divisé par le nombre d'adversaires.
    rivalry = 1.5,
    /// Difficile : fins de partie jouées par option, au choix du peuple et au déclin.
    pick_draws = 12.0,
    decline_draws = 8.0,
    /// Difficile : poids des modèles appris face aux fins de partie jouées (en nombre de fins de partie).
    pick_prior = 24.0,
    decline_prior = 16.0,
    /// Malus par jeton adverse actif voisin de la région visée : elle sera dure à garder.
    exposure = 0.0,
    /// Entrée d'un peuple neuf : modèle appris (1) ou rendement immédiat (0) ; difficile : les
    /// `entry_tried` meilleures régions jouées `entry_draws` fois jusqu'au bout.
    entry_model = 1.0,
    entry_tried = 6.0,
    entry_draws = 8.0,
    /// Conquêtes du niveau difficile : modèle appris sur l'avance sur le meilleur adversaire.
    conquest_lead = 1.0,
    /// Conquêtes du niveau normal : modèle appris (1) ou rendement immédiat (0).
    conquest_model = 1.0,
    /// Placement de fin de tour du niveau normal : modèle de garde (1) ou racine de la menace (0).
    hold_model = 1.0,
    /// Choix du peuple du niveau normal : modèle appris (1) ou table de valeurs (0).
    pick_model = 1.0,
    /// Déclin du niveau normal : modèle appris (1) ou règle à seuils (0).
    decline_model = 1.0,
    /// Règle à seuils : au plus `decline_est` conquêtes possibles, au moins `decline_left` tours restants.
    decline_est = 1.0,
    decline_left = 3.0,
    /// Robustes : décliner en fin de tour avec au moins `stout_own` régions et au plus `stout_ready` jetons libres.
    stout_own = 3.0,
    stout_ready = 3.0,
    /// Coût minimal d'une région pour y lâcher le dragon avant les conquêtes ordinaires.
    dragon_min = 4.0,
}
