"""Solver Agent Module with V-A Based Emotion Perception

This module implements the main decision-making agent for human-agent negotiation.
The Solver Agent combines time-based and behavior-based strategies with Valence-Arousal
(V-A) based affective perception (P_E) to generate optimal counter-offers.

The agent implements:
- Time-dependent concession curves (Bézier curve)
- Opponent behavior modeling (sensitivity analysis via k-means clustering)
- V-A emotion-aware bidding adjustments using Differential Affective Feedback:
    P_E = P_sign × √(P_rot² + P_rad²)
  where:
    - P_sign = sgn(ΔV + ΔA): directional sign
    - P_rot = atan2(δ_det, δ_dot): rotational change (normalized by π)
    - P_rad = |‖Ẽ_t‖ - ‖Ẽ_{t-1}‖|: intensity change (normalized by √2)
- Awareness-weighted emotion integration

References:
    See Section 4 of the paper for detailed methodology.
"""

import copy
import math
import json
import typing as t

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from HATN import negoAction
from Agent.Agent_Emotion.emotion_controller import EmotionController
from Agent.Solver_Agent.uncertainty_module import UncertaintyModule


class EstimatedSensitivityCalculator:
    """
    Calculates opponent sensitivity class using k-means clustering on move patterns.
    
    The calculator classifies opponent behavior into sensitivity classes:
    - Standard: Balanced negotiation style
    - Silent: Minimal concessions
    - Selfish: Self-serving moves dominate
    - Fortunate: Win-win oriented moves
    - Concession: Cooperative, conceding style
    """
    
    def __init__(self, kmeans_config_path: str = None):
        """
        Initialize with pre-trained k-means model.
        
        Args:
            kmeans_config_path: Path to k-means configuration JSON file.
                               If None, uses default from Agent/Solver_Agent/kmeans.json
        """
        if kmeans_config_path is None:
            import os
            kmeans_config_path = os.path.join(
                os.path.dirname(__file__), "kmeans.json"
            )
        
        with open(kmeans_config_path) as f:
            jsonData = json.load(f)
            self.kmeans = KMeans().fit(np.random.rand(10, 6))
            self.kmeans._n_threads = _openmp_effective_n_threads()
            self.kmeans.cluster_centers_ = np.asarray(jsonData["centers"])
            self.kmeans.labels_ = np.asarray(jsonData["labels"])
            self.kmeans.n_iter_ = jsonData["n_iter"]
            self.kmeans.inertia_ = jsonData["inertia"]

    def get_opponent_moves_list(
        self, 
        target_preference_profile, 
        opponent_preference_profile, 
        target_history: t.List
    ) -> t.List[str]:
        """
        Classify moves between consecutive offers.
        
        Gets preference profiles of both sides and given offers by target as input. 
        Can calculate for both agent and opponent based on given input.
        
        Args:
            target_preference_profile: Preference profile to calculate moves for
            opponent_preference_profile: Other party's preference profile
            target_history: List of offers from the target side
            
        Returns:
            List of move labels: "silent", "nice", "fortunate", "unfortunate", 
            "concession", "selfish"
        """
        # Keep opponent bids utility list.
        human_bid_utility_list = []
        # Keep target's bids utility list.
        target_bid_utility_list = []
        # Iterate target's history.
        for offer in target_history:
            # Calculate offer's utility for opponent.
            opp_bid_utility = opponent_preference_profile.get_offer_utility(
                offer.get_bid(perspective="Human")
            )
            # Calculate offer's utility for target.
            target_bid_utility = target_preference_profile.get_offer_utility(
                offer.get_bid(perspective="Agent")
            )
            # Add opponent's utility to the list.
            human_bid_utility_list.append(opp_bid_utility)
            # Add target's utility to the list.
            target_bid_utility_list.append(target_bid_utility)

        # Keep target's moves in the list.
        target_move_list = []
        # Iterate through all offers.
        for i in range(len(target_history) - 1):
            # Calculate moves between 2 offers.
            target_delta = target_bid_utility_list[i + 1] - target_bid_utility_list[i]
            human_delta = human_bid_utility_list[i + 1] - human_bid_utility_list[i]
            
            if abs(target_delta) == 0 and abs(human_delta) == 0:
                target_move_list.append("silent")
            elif abs(target_delta) == 0 and abs(human_delta) > 0:
                target_move_list.append("nice")
            elif target_delta > 0 and human_delta > 0:
                target_move_list.append("fortunate")
            elif target_delta < 0 and human_delta < 0:
                target_move_list.append("unfortunate")
            elif target_delta < 0 and human_delta > 0:
                target_move_list.append("concession")
            else:
                target_move_list.append("selfish")

        return list(target_move_list)

    def get_sensitivity_rate(self, target_move_list: t.List[str]) -> t.Dict[str, float]:
        """
        Calculate move frequency distribution.
        
        Args:
            target_move_list: List of move labels
            
        Returns:
            Dictionary mapping move types to their frequency [0, 1]
        """
        sensitivity_rate_dict = {
            "silent": 0.0, "nice": 0.0, "fortunate": 0.0, 
            "unfortunate": 0.0, "concession": 0.0, "selfish": 0.0
        }
        for move in target_move_list:
            sensitivity_rate_dict[move] += 1.0 / len(target_move_list)
        return sensitivity_rate_dict

    def get_human_awareness(
        self, 
        agent_preference_profile, 
        human_preference_profile, 
        agentHistory: t.List, 
        humanHistory: t.List
    ) -> float:
        """
        Calculate opponent's awareness score (P_A).
        
        Measures how responsive the human is to agent's move changes.
        Bounded to [0.25, 0.75].
        
        Args:
            agent_preference_profile: Agent's utility space
            human_preference_profile: Estimated human utility space
            agentHistory: Agent's offer history
            humanHistory: Human's offer history
            
        Returns:
            Awareness score in [0.25, 0.75]
        """
        human_moves_list = self.get_opponent_moves_list(
            human_preference_profile, agent_preference_profile, humanHistory
        )
        agent_moves_list = self.get_opponent_moves_list(
            agent_preference_profile, human_preference_profile, agentHistory
        )

        human_awareness = 0
        count = 0
        
        for i in range(1, len(human_moves_list) - 1):
            if agent_moves_list[i] != agent_moves_list[i - 1]:
                count += 1
            if (human_moves_list[i + 1] != human_moves_list[i]) and (
                agent_moves_list[i] != agent_moves_list[i - 1]
            ):
                human_awareness += 1

        try:
            human_awareness = human_awareness / (count * 1.0)
        except:
            human_awareness = 0

        # Bound to [0.25, 0.75]
        if human_awareness > 0.75:
            human_awareness = 0.75
        if human_awareness < 0.25:
            human_awareness = 0.25

        return human_awareness

    def get_sensitivity_index(
        self, 
        target_preference_profile, 
        opponent_preference_profile, 
        target_history: t.List
    ) -> int:
        """
        Get sensitivity class index using k-means prediction.
        
        Args:
            target_preference_profile: Target's utility space
            opponent_preference_profile: Opponent's estimated utility space
            target_history: List of target's offers
            
        Returns:
            Cluster index (0-4) representing sensitivity class
        """
        move_list = self.get_opponent_moves_list(
            target_preference_profile, opponent_preference_profile, target_history
        )
        sensitivity_rate_dict = self.get_sensitivity_rate(move_list)
        sensitivity_vector = np.array([[
            sensitivity_rate_dict["silent"], 
            sensitivity_rate_dict["nice"], 
            sensitivity_rate_dict["fortunate"], 
            sensitivity_rate_dict["unfortunate"], 
            sensitivity_rate_dict["concession"], 
            sensitivity_rate_dict["selfish"]
        ]])
        return self.kmeans.predict(sensitivity_vector)[0]


class SolverAgent:
    """
    Main negotiation agent implementing V-A based emotion-aware bidding strategy.
    
    The Solver Agent combines multiple strategy components:
    - Time-based component: Bézier curve concession
    - Behavior-based component: Reciprocity-based adjustments
    - Emotion effect (P_E): Differential Affective Feedback
    - Awareness (P_A): Opponent model confidence
    
    Key Formula (P_E calculation - Differential Affective Feedback):
        P_sign = sgn(ΔV + ΔA)  # Directional sign
        P_rot = atan2(δ_det, δ_dot)  # Rotational change
        P_rad = |‖Ẽ_t‖ - ‖Ẽ_{t-1}‖|  # Intensity change
        P_E = P_sign × √(P_rot² + P_rad²)  # Final signal
        
        where δ_det = det(Ẽ_{t-1}, Ẽ_t) and δ_dot = Ẽ_{t-1} · Ẽ_t
    
    Final target utility:
        BB_final = BB_base + (awareness² × P_E)
        target = (1 - t²) × BB_final + t² × time_based
    
    Attributes:
        utility_space: Agent's preference model
        time_controller: Deadline manager
        agent_history: List of agent's past offers
        opponent_history: List of opponent's past offers
        human_awareness: P_A - confidence in opponent model [0.25, 0.75]
        previous_arousal: Last round's arousal value
        previous_valance: Last round's valence value
    """
    
    def __init__(self, utility_space, time_controller, action_factory=None):
        """
        Initialize the Solver Agent.
        
        Args:
            utility_space: Agent's utility space configuration
            time_controller: Negotiation deadline controller
            action_factory: Factory for creating negotiation actions (optional)
        """
        self.utility_space = utility_space
        self.estimated_opponent_preference = copy.deepcopy(self.utility_space)
        self.action_factory = action_factory
        self.time_controller = time_controller
        self.estimated_sensitivity_calculator = EstimatedSensitivityCalculator()
        self.uncertainty_module = UncertaintyModule(utility_space)

        self.agent_history = []
        self.opponent_history = []
        self.sensitivity_class = {
            0: "Standart",
            1: "Silent",
            2: "Selfish",
            3: "Fortunate",
            4: "Concession",
        }
        self.mood_evaluations = {
            "Surprise": 0.33,
            "Happiness": 0.165,
            "Neutral": 0,
            "Disgust": 0,
            "Fear": 0,
            "Anger": -0.165,
            "Sadness": -0.33,
        }
        self.human_awareness = 0.5  # P_A - will be updated after 8 rounds
        self.silent_nash_index = 0

        self.emotion_distance = 0
        self.sensitivity_class_list = []
        self.my_prev_util = 0
        
        # V-A state tracking for P_E calculation (Differential Affective Feedback)
        self.previous_arousal = 0
        self.previous_valance = 0
        self.previous_magnitude = 0  # ‖Ẽ_{t-1}‖ for P_rad calculation

        # Initialize emotion controller (for agent mood feedback)
        self.emotion_controller = EmotionController(
            self.utility_space, self.time_controller
        )
        # Initialize previous sensitivity class as none
        self.previous_sensitivity_class = None

        # Bézier curve control points for time-based concession
        self.p0 = 0.9  # Initial aspiration
        self.p1 = 0.7  # First control point
        self.p2 = 0.4  # Target at deadline
        self.p3 = 0.5  # Behavior-based parameter

        # Weighted history for behavior-based component
        self.W = {
            1: [1],
            2: [0.25, 0.75],
            3: [0.11, 0.22, 0.66],
            4: [0.05, 0.15, 0.3, 0.5],
        }

        self.bid_frequencies = {}
        self.delta_multiplier = 1

    def time_based(self, t: float) -> float:
        """
        Calculate time-based target utility using Bézier curve.
        
        Args:
            t: Normalized time [0, 1]
            
        Returns:
            Target utility value
        """
        return (
            (1 - t) * (1 - t) * self.p0 
            + 2 * (1 - t) * t * self.p1 
            + t * t * self.p2
        )

    def behaviour_based(self) -> float:
        """
        Calculate behavior-based target utility using weighted opponent deltas.
        
        Uses exponentially weighted recent opponent utility changes to
        calculate reciprocal adjustment.
        
        Returns:
            Behavior-based target utility
        """
        t = self.time_controller.get_remaining_time()

        diff = [
            self.utility_space.get_offer_utility(self.opponent_history[i + 1]) 
            - self.utility_space.get_offer_utility(self.opponent_history[i])
            for i in range(len(self.opponent_history) - 1)
        ]

        if len(diff) > len(self.W):
            diff = diff[-len(self.W):]

        delta = sum([u * w for u, w in zip(diff, self.W[len(diff)])]) * self.delta_multiplier

        mu = self.p3 + self.p3 * t

        previous_agent_offer_utility = self.utility_space.get_offer_utility(
            self.agent_history[-1]
        )

        target_utility = previous_agent_offer_utility - (
            (1 - (self.human_awareness ** 2)) * (mu * delta)
        )

        print("BB TARGET: ", target_utility, previous_agent_offer_utility, mu, delta)

        return target_utility

    def check_acceptance(
        self, 
        final_target_utility: float, 
        human_offer_utility: float
    ) -> t.Tuple[bool, t.Tuple]:
        """
        Check if human offer should be accepted.
        
        Args:
            final_target_utility: Agent's target utility
            human_offer_utility: Utility of human's offer for agent
            
        Returns:
            Tuple of (should_accept, (Accept_action, mood)) or (False, ())
        """
        if final_target_utility < human_offer_utility:
            self.my_prev_util = final_target_utility
            return True, (negoAction.Accept(), "Happy")
        return False, ()

    def receive_offer(
        self, 
        human_offer, 
        predictions: t.Dict[str, float], 
        normalized_predictions: t.Dict[str, float]
    ) -> t.Tuple:
        """
        Process incoming human offer and generate counter-offer using V-A predictions.
        
        This is the main decision-making function that:
        1. Updates opponent history
        2. Extracts V-A values from normalized predictions
        3. Calculates P_E (emotion effect) from V-A differentials
        4. Applies time-based and behavior-based components
        5. Integrates P_E weighted by awareness²
        6. Generates optimal counter-offer
        
        P_E Calculation (Differential Affective Feedback):
            P_sign = sgn(delta_v + delta_a)  # Directional sign
            P_rot = atan2(delta_det, delta_dot) / π  # Normalized rotation
            P_rad = |current_magnitude - previous_magnitude| / √2  # Normalized radial
            P_E = P_sign × min(√(P_rot² + P_rad²), 1.0)  # Bounded signal
        
        Args:
            human_offer: The offer received from the human participant
            predictions: Raw V-A predictions from CLIFER/FC module
                        Format: {"Arousal": float, "Valance": float, "Max_A": float, ...}
            normalized_predictions: Session-normalized V-A values
                        Format: {"Arousal": float, "Valance": float}
        
        Returns:
            Tuple of (action, mood):
            - action: Accept() or Offer(bid) negotiation action
            - mood: Agent's emotional response label
        """
        human_offer_utility = self.utility_space.get_offer_utility(
            human_offer.get_bid(perspective="Agent")
        )
        emotion_value = 0

        self.opponent_history.append(human_offer)

        current_time = self.time_controller.get_remaining_time()
        time_based_target_utility = self.time_based(current_time)

        behavior_based_target_utility = 0
        behavior_based_utility = 0

        final_target_utility = time_based_target_utility
        
        # Extract V-A from normalized predictions
        arousal = normalized_predictions["Arousal"]
        valance = normalized_predictions["Valance"]
        sensitivity_class = ""

        if len(self.opponent_history) > 1:
            # ========== P_E CALCULATION (Differential Affective Feedback) ==========
            # Calculate V-A differentials
            delta_v = valance - self.previous_valance
            delta_a = arousal - self.previous_arousal
            
            # Current state magnitude ‖Ẽ_t‖
            current_magnitude = math.sqrt(valance ** 2 + arousal ** 2)
            
            # P_sign: Directional sign from net V-A change (Eq. 6)
            # Alpha coefficient for weighting arousal contribution (default: 1.0)
            alpha = 1.0  # Adjustable: set alpha < 1.0 to dampen arousal influence
            p_sign = 1 if (delta_v + alpha * delta_a) >= 0 else -1
            
            # P_rot: Rotation component using atan2 (Eq. 7)
            # δ_dot = Ẽ_{t-1} · Ẽ_t (dot product)
            # δ_det = det(Ẽ_{t-1}, Ẽ_t) (2D cross product / determinant)
            delta_dot = (self.previous_valance * valance + 
                         self.previous_arousal * arousal)
            delta_det = (self.previous_valance * arousal - 
                         self.previous_arousal * valance)
            p_rot = math.atan2(delta_det, delta_dot)
            
            # P_rad: Radial/intensity change component (Eq. 7)
            p_rad = abs(current_magnitude - self.previous_magnitude)
            
            # Normalize components to [0, 1]
            p_rot_normalized = abs(p_rot) / math.pi  # max rotation is π
            p_rad_normalized = p_rad / math.sqrt(2)  # max magnitude change for normalized vectors
            
            # Combined magnitude with clipping (Eq. 8)
            p_mag = min(math.sqrt(p_rot_normalized ** 2 + p_rad_normalized ** 2), 1.0)
            
            # Final P_E: sign × magnitude
            emotion_value = p_sign * p_mag
            
            # Update magnitude for next round
            self.previous_magnitude = current_magnitude
            # =====================================================================

            behavior_based_utility = self.behaviour_based()
            
            # Integrate P_E weighted by awareness²
            behavior_based_target_utility = behavior_based_utility + (
                (self.human_awareness ** 2) * emotion_value
            )
            behavior_based_target_utility = min(behavior_based_target_utility, 1.0)
            
            # Mix time-based and behavior-based components
            final_target_utility = (
                (1 - current_time ** 2) * behavior_based_target_utility 
                + (current_time ** 2) * time_based_target_utility
            )

            if len(self.opponent_history) > 8:
                # Calculate awareness based on estimated opponent preference
                self.human_awareness = (
                    self.estimated_sensitivity_calculator.get_human_awareness(
                        self.utility_space,
                        self.estimated_opponent_preference,
                        self.agent_history,
                        self.opponent_history,
                    )
                )
                # Get sensitivity class and update parameters
                sensitivity_class_index = (
                    self.estimated_sensitivity_calculator.get_sensitivity_index(
                        self.utility_space,
                        self.estimated_opponent_preference,
                        self.opponent_history,
                    )
                )

                sensitivity_class = self.sensitivity_class[sensitivity_class_index]
                self.update_with_sensitivity_class(sensitivity_class)

                # Generate offer (used for logging)
                generated_offer = self.utility_space.get_offer_below_utility(
                    final_target_utility
                )
                generated_offer_utility = self.utility_space.get_offer_utility(
                    generated_offer
                )

        generated_offer = self.utility_space.get_offer_below_utility(final_target_utility)

        # Update V-A state for next round
        self.previous_arousal = arousal
        self.previous_valance = valance

        generated_offer_utility = self.utility_space.get_offer_utility(generated_offer)
        self.my_prev_util = generated_offer_utility

        if generated_offer_utility < human_offer_utility:
            mood = "Happy"
            self.agent_history.append(generated_offer)
            self.agent_history.append(negoAction.Accept())
        else:
            mood, _ = self.emotion_controller.get_emotion(human_offer, 0.5)
            self.agent_history.append(generated_offer)

        # Log negotiation state (for analysis)
        self._log_round(
            "Human", human_offer, human_offer_utility, current_time,
            behavior_based_utility, behavior_based_target_utility,
            emotion_value, time_based_target_utility, final_target_utility,
            predictions, normalized_predictions, sensitivity_class
        )
        self._log_round(
            "Agent", generated_offer, generated_offer_utility, current_time,
            behavior_based_utility, behavior_based_target_utility,
            emotion_value, time_based_target_utility, final_target_utility,
            predictions, normalized_predictions, sensitivity_class
        )

        print("T: ", current_time)
        print("Final: ", final_target_utility, " Behavior Based: ", 
              behavior_based_target_utility, " Time Based: ", time_based_target_utility)
        print("ITEM SENT: ", self.agent_history[-1], " - UTIL: ", generated_offer_utility)
        
        return self.agent_history[-1], mood

    def _log_round(
        self, logger_type, offer, utility, scaled_time,
        behavior_based, behavior_based_final, pe, time_based, 
        final_utility, predictions, normalized_predictions, sensitivity_class
    ):
        """
        Log round data for analysis.
        
        Note: In production, this would call LoggerNew.log_solver().
        For reproducibility, logging is abstracted here.
        """
        log_data = {
            "logger": logger_type,
            "offer": offer.get_bid(perspective="Agent") if hasattr(offer, 'get_bid') else offer,
            "agent_utility": utility,
            "scaled_time": scaled_time,
            "behavior_based": behavior_based,
            "behavior_based_final": behavior_based_final,
            "pe": pe,  # P_E: emotion effect
            "pa": self.human_awareness,  # P_A: awareness
            "time_based": time_based,
            "final_utility": final_utility,
            "predictions": predictions,
            "normalized_predictions": normalized_predictions,
            "sensitivity_class": sensitivity_class
        }
        # LoggerNew.log_solver(log_data)  # Uncomment for production
        pass

    def receive_negotiation_over(
        self, 
        participant_name: str, 
        session_number: str, 
        type: str
    ) -> t.Dict[str, int]:
        """
        Handle negotiation completion.
        
        Args:
            participant_name: Name of the participant
            session_number: Session identifier
            type: How negotiation ended ("agent", "human", "timeout")
            
        Returns:
            Dictionary of mood counts during negotiation
        """
        num_of_emotions = self.emotion_controller.get_num_of_emotions()
        return num_of_emotions

    def update_with_sensitivity_class(self, sensitivity_class: str):
        """
        Adjust strategy parameters based on detected sensitivity class.
        
        Args:
            sensitivity_class: Detected opponent behavior class
        """
        if sensitivity_class == "Fortunate" and self.previous_sensitivity_class != "Fortunate":
            self.p1 = self.p1 - 0.2
        elif sensitivity_class == "Standart" and self.previous_sensitivity_class != "Standart":
            pass
        elif sensitivity_class == "Silent" and self.previous_sensitivity_class != "Silent":
            pass
        elif sensitivity_class == "Selfish" and self.previous_sensitivity_class != "Selfish":
            self.delta_multiplier = 1.5
        elif sensitivity_class == "Concession" and self.previous_sensitivity_class != "Concession":
            self.p1 = self.p1 + 0.2
            self.p2 = self.p2 + 0.2
        self.previous_sensitivity_class = sensitivity_class
