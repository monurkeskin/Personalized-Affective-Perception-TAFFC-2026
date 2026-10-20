from HATN.utilitySpace import UtilitySpace
from HATN.negoTime import NegotiationTimeController


class EmotionController:
    def __init__(self, utility_space, time_controller):
        # Set emotion counts for agent.
        self.num_of_emotions = {
            "Frustrated": 0,
            "Annoyed": 0,
            "Dissatisfied": 0,
            "Neutral": 0,
            "Convinced": 0,
            "Content": 0,
            "Worried": 0,
        }
        # Set utility space.
        self.utility_space = utility_space
        # Set time controller.
        self.time_controller = time_controller
        # Keep human's previous offers to compare with current offer.
        self.opponent_previous_offers = []
        # Human's previous offer utility.
        self.opponent_previous_offer_utility = None
        # First warning flag for deadline.
        self.did_first_warn = False
        # Second warning flag for deadline.
        self.did_second_warn = False
        # Third warning flag for deadline.
        self.did_third_warn = False
        # Keep file names for returning with order.
        self.frustrated_files = [
            "frustrated_1",
            "frustrated_3",
            "frustrated_2",
            "frustrated_4",
        ]
        self.annoyed_files = [
            "annoyed_1",
            "annoyed_3",
            "annoyed_6",
            "annoyed_2",
            "annoyed_4",
            "annoyed_7",
            "annoyed_5",
        ]
        self.dissatisfied_files = [
            "dissatisfied_1",
            "dissatisfied_3",
            "dissatisfied_6",
            "dissatisfied_2",
            "dissatisfied_4",
            "dissatisfied_7",
            "dissatisfied_5",
            "dissatisfied_8",
        ]
        self.convinced_files = [
            "convinced_1",
            "convinced_2",
            "convinced_3",
            "convinced_4",
        ]
        self.content_files = ["content_1", "content_2"]
        self.neutral_files = ["neutral_1", "neutral_2"]

    def get_emotion(self, human_offer, threshold):
        """
        This function gets offer of the opponent's and lower threshold of the current tactic as input.
        Return robot emotion and emotion method to call and updates emotion counts.
        """
        emotion = None
        # Get human's offer as list, so that we can compare with previous ones.
        human_offer_list = human_offer.get_bid()
        # Set default emotion and emotion file to none.
        emotion = None
        emotion_file = None

        # If 6 minutes remaining.
        if self.time_controller.get_current_time() > 0.6 and not self.did_first_warn:
            self.num_of_emotions["Worried"] += 1
            self.did_first_warn = True
            emotion = "Worried"
            emotion_file = "worried_1"
        # If 4 minutes remaining.
        elif (
            self.time_controller.get_current_time() > 0.73 and not self.did_second_warn
        ):
            self.num_of_emotions["Worried"] += 1
            self.did_second_warn = True
            emotion = "Worried"
            emotion_file = "worried_2"
        # If 2 minutes remaining.
        elif self.time_controller.get_current_time() > 0.86 and not self.did_third_warn:
            self.num_of_emotions["Worried"] += 1
            self.did_third_warn = True
            emotion = "Worried"
            emotion_file = "worried_3"
        # If offer utility below reservation value or time is too close and utility is below 0.5
        elif (self.utility_space.get_offer_utility(human_offer) < 0.4) or (
            self.utility_space.get_offer_utility(human_offer) < 0.5
            and self.time_controller.get_current_time() > 0.73
        ):
            emotion_file_idx = self.num_of_emotions["Frustrated"] % len(
                self.frustrated_files
            )
            emotion_file = self.frustrated_files[emotion_file_idx]
            self.num_of_emotions["Frustrated"] += 1
            emotion = "Frustrated"
        # Check whether we have previous utility or not, so that we can compare with previous offers & utilities.
        elif not self.opponent_previous_offer_utility == None:
            # Calculate the utility diff between this and previous offer.
            utility_delta = (
                self.utility_space.get_offer_utility(human_offer)
                - self.opponent_previous_offer_utility
            )

            if (
                len(self.opponent_previous_offers) >= 2
                and human_offer_list == self.opponent_previous_offers[-1]
                and human_offer_list == self.opponent_previous_offers[-2]
            ):
                emotion_file_idx = self.num_of_emotions["Frustrated"] % len(
                    self.frustrated_files
                )
                emotion_file = self.frustrated_files[emotion_file_idx]
                self.num_of_emotions["Frustrated"] += 1
                emotion = "frustrated"
            elif human_offer_list == self.opponent_previous_offers[-1]:
                emotion_file_idx = self.num_of_emotions["Annoyed"] % len(
                    self.annoyed_files
                )
                emotion_file = self.annoyed_files[emotion_file_idx]
                self.num_of_emotions["Annoyed"] += 1
                emotion = "Annoyed"
            elif utility_delta == 0:
                emotion_file_idx = self.num_of_emotions["Neutral"] % len(
                    self.neutral_files
                )
                emotion_file = self.neutral_files[emotion_file_idx]
                self.num_of_emotions["Neutral"] += 1
                emotion = "Neutral"
            elif 0 < utility_delta and utility_delta <= 0.25:
                emotion_file_idx = self.num_of_emotions["Convinced"] % len(
                    self.convinced_files
                )
                emotion_file = self.convinced_files[emotion_file_idx]
                self.num_of_emotions["Convinced"] += 1
                emotion = "Convinced"
            elif 0.25 < utility_delta:
                emotion_file_idx = self.num_of_emotions["Content"] % len(
                    self.content_files
                )
                emotion_file = self.content_files[emotion_file_idx]
                self.num_of_emotions["Content"] += 1
                emotion = "Content"
            elif -0.25 <= utility_delta < 0:
                emotion_file_idx = self.num_of_emotions["Dissatisfied"] % len(
                    self.dissatisfied_files
                )
                emotion_file = self.dissatisfied_files[emotion_file_idx]
                self.num_of_emotions["Dissatisfied"] += 1
                emotion = "Dissatisfied"
            elif utility_delta < -0.25:
                emotion_file_idx = self.num_of_emotions["Annoyed"] % len(
                    self.annoyed_files
                )
                emotion_file = self.annoyed_files[emotion_file_idx]
                self.num_of_emotions["Annoyed"] += 1
                emotion = "Annoyed"

        # Set offer's utility as previous after done.
        self.opponent_previous_offer_utility = self.utility_space.get_offer_utility(
            human_offer
        )
        # Append to the offer history.
        self.opponent_previous_offers.append(human_offer_list)
        # Return the robot action.
        return emotion, emotion_file

    def is_close_to_agreement(self, human_offer, threshold):
        human_offer_utility = self.utility_space.get_offer_utility(human_offer)
        if human_offer_utility >= threshold:
            return True
        else:
            return False

    def get_num_of_emotions(self):
        return self.num_of_emotions
