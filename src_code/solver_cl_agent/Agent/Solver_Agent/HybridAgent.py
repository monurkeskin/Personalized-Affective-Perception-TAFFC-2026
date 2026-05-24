import copy
from HATN import negoAction
from Agent.Agent_Emotion.emotion_controller import EmotionController
from Agent.Solver_Agent.estimated_sensitivity_calculator import (
    EstimatedSensitivityCalculator,
)

# Import helper classes.
from Agent.Solver_Agent.time_based import kek
from Agent.Solver_Agent.behavior_based import bb
from Agent.Solver_Agent.uncertainty_module import UncertaintyModule


class HybridAgent:
    def __init__(self, utility_space, time_controller):
        self.utility_space = utility_space
        self.estimatedOpponentPreference = copy.deepcopy(self.utility_space)
        self.time_controller = time_controller
        self.estimated_sensitivity_calculator = EstimatedSensitivityCalculator()

        self.agent_history = []
        self.opponent_history = []
        self.sensitivity_class = {
            0: "Standart",
            1: "Silent",
            2: "Selfish",
            3: "Fortunate",
            4: "Concession",
        }
        self.emotion_evaluations = {
            "Surprise": 0.33,
            "Happiness": 0.165,
            "Neutral": 0,
            "Disgust": 0,
            "Fear": 0,
            "Anger": -0.165,
            "Sadness": -0.33,
        }
        self.awareness = 0.5  # Will fix.
        self.silent_nash_index = 0  #

        self.sensitivity_class_list = []

        self.my_prev_util = 0

        self.behavior_based = bb()
        self.time_based = kek()
        self.uncertaintyModule = UncertaintyModule(utility_space)

        # Initialize emotion controller.
        self.emotion_controller = EmotionController(
            self.utility_space, self.time_controller
        )
        # Initialize previous sensitivity class as none.
        self.previous_sensitivity_class = None

    def receive_offer(self, human_offer, valance_arousal_predictions):
        """
        This function is called when the agent receives offer with ***emotion_recording=True.
        """

        print("Hybrid valance arousal", valance_arousal_predictions)

        self.opponent_history.append(human_offer)
        self.estimatedOpponentPreference = (
            self.uncertaintyModule.estimate_opponent_preferences(self.opponent_history)
        )

        current_time = self.time_controller.get_current_time()
        # print("current time:", current_time)

        kek_target_utility = self.time_based.get_target_utility(current_time)

        final_target_utility = kek_target_utility

        self.sensitivity_class_list.append(" ")

        if len(self.opponent_history) > 1:
            # Calculate target utility based on mixed model (time-based and behavior-based).
            utilityDiff = self.utility_space.get_offer_utility(
                self.opponent_history[-1]
            ) - self.utility_space.get_offer_utility(self.opponent_history[-2])
            bb_target_utility = self.behavior_based.get_target_utility(
                utilityDiff, self.my_prev_util, current_time, 0
            )
            final_target_utility = (1 - current_time**2) * bb_target_utility + (
                current_time**2
            ) * kek_target_utility

            if final_target_utility < self.utility_space.get_offer_utility(human_offer):
                return negoAction.Accept(), "Happy", "accept"
            else:
                # Get closest offer to that utility.
                generated_offer = self.utility_space.get_offer_below_utility(
                    final_target_utility
                )

                threshold = self.utility_space.get_offer_utility(generated_offer) * 0.9
                agent_emotion, agent_emotion_file = self.emotion_controller.get_emotion(
                    human_offer, threshold
                )

                self.agent_history.append(negoAction.Offer(generated_offer))
                self.my_prev_util = final_target_utility

                return self.agent_history[-1], agent_emotion, agent_emotion_file
        else:
            if final_target_utility < self.utility_space.get_offer_utility(human_offer):
                return negoAction.Accept(), "Happy", "accept"
            else:
                # Get closest offer to that utility.
                generated_offer = self.utility_space.get_offer_below_utility(
                    final_target_utility
                )

                threshold = self.utility_space.get_offer_utility(generated_offer) * 0.9
                agent_emotion, agent_emotion_file = self.emotion_controller.get_emotion(
                    human_offer, threshold
                )

                self.agent_history.append(negoAction.Offer(generated_offer))
                self.my_prev_util = final_target_utility

                return self.agent_history[-1], agent_emotion, agent_emotion_file

    def update_with_sensitivity_class(self, sensitivity_class):
        if (
            sensitivity_class == "Fortunate"
            and self.previous_sensitivity_class != "Fortunate"
        ):
            P1 = self.time_based.P1
            P1 = P1 - 0.2
            self.time_based.update_P1(P1)
        elif (
            sensitivity_class == "Standart"
            and self.previous_sensitivity_class != "Standart"
        ):
            pass
        elif (
            sensitivity_class == "Silent"
            and self.previous_sensitivity_class != "Silent"
        ):
            pass
        elif (
            sensitivity_class == "Selfish"
            and self.previous_sensitivity_class != "Selfish"
        ):
            self.behavior_based.set_delta_multiplier(1.5)
        elif (
            sensitivity_class == "Concession"
            and self.previous_sensitivity_class != "Concession"
        ):
            P1 = self.time_based.P1
            P2 = self.time_based.P2
            P1 = P1 + 0.2
            P2 = P2 + 0.2
            self.time_based.update_P1(P1)
            self.time_based.update_P2(P2)
        self.previous_sensitivity_class = sensitivity_class

    def receive_accept(self, human_action):
        """
        It is called at the end of the acceptance.
        """
        num_of_emotions = self.emotion_controller.get_num_of_emotions()
        return num_of_emotions

    def receive_end_of_deadline(self):
        """
        It is called when negotiation ends without an agreement.
        """
        num_of_emotions = self.emotion_controller.get_num_of_emotions()
        return num_of_emotions
