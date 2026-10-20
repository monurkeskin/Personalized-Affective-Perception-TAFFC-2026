from Agent_Interaction_Models.robot_mobile_action import RobotMobileAction
from Agent_Interaction_Models.robot_stiffness_action import RobotStiffnessAction
from Agent_Interaction_Models.cli import CLI
from Human_Interaction_Models.speech_controller import SpeechController
from Human_Interaction_Models.human_cli import HumanCLI
from Human_Interaction_Models.camera_controller import CameraController
from Agent.Mimic_Agent import MimicAgent
from Agent.TSBT_Agent import TSBTAgent
from Agent.Solver_Agent.Solver_Agent import SolverAgent
from Agent.Solver_Agent.HybridAgent import HybridAgent
from utility_space_controller import UtilitySpaceController
from utilitySpace import UtilitySpace
from negoTime import NegotiationTimeController
from Logger.logger import Logger
from negoHistory import NegotiationHistory
from HATN import negoAction
import time
from HATN.negoAction import Accept, Offer

from GUI.negotiation_gui import NegotiationGUI


class HAT:
    def __init__(
        self,
        participant_name,
        human_interaction_type,
        agent_interaction_type,
        agent_type,
        agent_preference_file,
        human_preference_file,
        domain_file,
        deadline,
        session_number,
        log_file_path,
        domain,
    ):
        """
        Does stuff.
        """
        self.agent = None
        self.is_first_turn = True
        self.logger = Logger(
            participant_name, agent_type, agent_interaction_type, log_file_path, domain
        )
        self.time_controller = NegotiationTimeController(deadline)
        self.session_number = session_number
        self.camera_controller = CameraController(self.session_number)
        self.negotiation_setup(
            human_interaction_type,
            agent_interaction_type,
            agent_type,
            agent_preference_file,
            human_preference_file,
            domain_file,
        )

    def negotiation_setup(
        self,
        human_interaction_type,
        agent_interaction_type,
        agent_type,
        agent_preference_file,
        human_preference_file,
        domain_file,
    ):
        self.set_utility_space(
            agent_preference_file, human_preference_file, domain_file
        )
        self.negotiation_gui = NegotiationGUI(None, self.human_utility_space)
        self.set_agent_type(agent_type)
        self.set_agent_interaction_type(agent_interaction_type)
        self.set_human_interaction_type(human_interaction_type, domain_file)
        # Camera controller join etc.
        self.camera_controller.join()

    def set_human_interaction_type(self, human_interaction_type, domain_file):
        if human_interaction_type == "Microphone":
            self.human_interaction_controller = SpeechController(
                self.human_utility_space,
                domain_file,
                self.negotiation_gui,
                self.time_controller,
            )
        elif human_interaction_type == "Human-CLI":
            self.human_interaction_controller = HumanCLI(
                self.human_utility_space,
                domain_file,
                self.negotiation_gui,
                self.time_controller,
            )
        else:
            raise "Invalid human interaction type."

    def set_agent_interaction_type(self, agent_interaction_type):
        if agent_interaction_type == "Robot-Stiffness":
            self.agent_interaction_controller = RobotStiffnessAction()
        elif agent_interaction_type == "Robot-Mobile":
            self.agent_interaction_controller = RobotMobileAction()
        elif agent_interaction_type == "Agent-CLI":
            self.agent_interaction_controller = CLI()
        else:
            raise "Invalid agent interaction type."

    def set_utility_space(
        self, agent_preference_file, human_preference_file, domain_file
    ):
        self.utility_space_controller = UtilitySpaceController(domain_file)
        self.agent_utility_space = UtilitySpace(agent_preference_file)
        self.human_utility_space = UtilitySpace(human_preference_file)
        self.nego_history = NegotiationHistory(
            self.utility_space_controller,
            self.agent_utility_space,
            self.human_utility_space,
        )

    def set_agent_type(self, agent_type):
        if agent_type == "Mimic":
            self.agent = MimicAgent(self.agent_utility_space, self.time_controller)
        elif agent_type == "TSBT":
            self.agent = TSBTAgent(self.agent_utility_space, self.time_controller)
        elif agent_type == "Solver":
            self.agent = SolverAgent(self.agent_utility_space, self.time_controller)
        elif agent_type == "Hybrid":
            self.agent = HybridAgent(self.agent_utility_space, self.time_controller)
        else:
            raise "Invalid agent type."

    def negotiate(self):
        # Initialize negotiation time.
        self.time_controller.start_negotiation_time()
        # Use robot's greet function.
        self.agent_interaction_controller.greet()
        # Check if the domain is resource allocation domain or not. It has extra steps.
        self.do_normal_nego()

    def set_negotiation_gui(self, agent_offer_to_human):
        offer = ""
        for index, value in enumerate(agent_offer_to_human):
            offer += (
                str(value)
                + " "
                + self.human_utility_space.issue_index_to_name[index]
                + " "
            )

        self.negotiation_gui.update_offer("Caduceus' offer: " + offer)
        self.negotiation_gui.update_offer_utility(
            str(
                int(
                    self.human_utility_space.get_offer_utility(agent_offer_to_human)
                    * 100
                )
            )
        )
        self.negotiation_gui.update_time(
            round(self.time_controller.get_current_time(), 2) * 600
        )

    def send_agent_offer_to_human(self, agent_offer_to_human, agent_emotion_file):
        if isinstance(self.agent_interaction_controller, CLI):
            # Convert it to dict as we need to say 3 apples etc.
            agent_offer_to_human_dict = (
                self.utility_space_controller.convert_offer_list_to_dict(
                    agent_offer_to_human
                )
            )
            print("Agent Offer To Human:", agent_offer_to_human_dict)
        else:
            # Convert it to dict as we need to say 3 apples etc.
            agent_offer_to_human_dict = (
                self.utility_space_controller.convert_offer_list_to_dict(
                    agent_offer_to_human
                )
            )
            # Call gesture function from given agent_emotion as string.
            if agent_emotion_file != None:
                getattr(self.agent_interaction_controller, agent_emotion_file)()
            # Send offer to the user.
            # offer_message = "\\pau=500\\".join(['%s %s' % (value, key) for (key, value) in agent_offer_to_human_dict.items()])
            # offer_message = ""
            # for (key, value) in agent_offer_to_human_dict.items():
            # 	offer_message += str(value) + " " + str(key) + " "
            # self.agent_interaction_controller.say("Here is my offer.")
            # self.agent_interaction_controller.say(str(offer_message))
            print("Agent Offer To Human:", agent_offer_to_human_dict)
            self.agent_interaction_controller.tell_offer(agent_offer_to_human_dict)
            self.agent_interaction_controller.send_after_offer_sentence()

    def end_negotiation_with_no_result(self, agent_num_of_emotions):
        self.nego_history.set_sensitivity_predictions(self.agent.sensitivity_class_list)
        self.nego_history.set_sentences(
            self.human_interaction_controller.human_sentences
        )
        # Send end of the negotiation action to the human.
        self.agent_interaction_controller.leave_negotiation()
        # Get offer history as df list.
        offer_df_list = self.nego_history.extract_history_to_df()
        # Log offer history with session number as sheet name.
        self.logger.log_offer_history(self.session_number, offer_df_list)
        # Get human's awareness.
        human_awareness = self.nego_history.get_human_awareness()
        # Get total number of offers.
        total_offers = self.nego_history.get_offer_count()
        # Get human's sensitivity rate dictionary.
        human_sensitivity_dict = self.nego_history.get_human_sensitivity_rates()
        # Log negotiation summary with agent emotions.
        self.logger.log_negotiation_summary(
            agent_num_of_emotions,
            human_sensitivity_dict,
            human_awareness,
            total_offers,
            1,
            is_agreement=False,
            agent_score=0,
            user_score=0,
        )
        # Stop camera and set next to controller if its first session, otherwise exit the application.
        if self.camera_controller and self.session_number == 1:
            self.camera_controller.stop_recording()
            self.camera_controller.next()
        else:
            self.camera_controller.stop_recording()
            self.camera_controller.exit()

    def end_negotiation_with_acceptance(self, agent_num_of_emotions):
        self.nego_history.set_sensitivity_predictions(self.agent.sensitivity_class_list)
        self.nego_history.set_sentences(
            self.human_interaction_controller.human_sentences
        )
        # Get offer history as df list.
        offer_df_list = self.nego_history.extract_history_to_df()
        # Log offer history with session number as sheet name.
        self.logger.log_offer_history(self.session_number, offer_df_list)
        # Get last offer of the negotiation (accepted one).
        last_offer = self.nego_history.get_last_offer()
        # Get human's awareness.
        human_awareness = self.nego_history.get_human_awareness()
        # Get total number of offers.
        total_offers = self.nego_history.get_offer_count()
        # Get human's sensitivity rate dictionary.
        human_sensitivity_dict = self.nego_history.get_human_sensitivity_rates()
        print("acceptance")
        # Log negotiation summary with agent emotions.
        self.logger.log_negotiation_summary(
            agent_num_of_emotions,
            human_sensitivity_dict,
            human_awareness,
            total_offers,
            last_offer[4],
            True,
            last_offer[2],
            last_offer[3],
        )
        print("Logged")
        # Stop camera and set next to controller if its first session, otherwise exit the application.
        if self.camera_controller and self.session_number == 1:
            print("Next")
            self.camera_controller.next()
        else:
            print("Exit")
            self.camera_controller.exit()

    def do_normal_nego(self):
        # While negotiation continues.
        while True:

            if self.is_first_turn:
                valance_arousal_predictions = {"Valance": 0, "Arousal": 0}
                self.is_first_turn = False
            else:
                valance_arousal_predictions = self.camera_controller.stop_recording()
            print("Valance arousal predictions: ", valance_arousal_predictions)

            human_action = (
                self.human_interaction_controller.get_human_action()
            )  # User's offer (items to the agent).
            # Check if human accepts the offer.
            if isinstance(human_action, negoAction.Accept):
                if self.time_controller.time_is_up():
                    agent_num_of_emotions = self.agent.receive_end_of_deadline()
                    self.end_negotiation_with_no_result(agent_num_of_emotions)
                    break
                else:
                    # Send end of the negotiation action to the human as acceptance.
                    self.agent_interaction_controller.say(
                        "Okay the negotiation is over."
                    )
                    agent_num_of_emotions = self.agent.receive_accept(human_action)
                    self.end_negotiation_with_acceptance(agent_num_of_emotions)
                    break
				
            if self.time_controller.time_is_up():
                agent_num_of_emotions = self.agent.receive_end_of_deadline()
                self.end_negotiation_with_no_result(agent_num_of_emotions)
                break

            # Get user's items that gives to the agent as dict.
            human_offer_dict = human_action.get_bid()
            # Convert dict to list and set human's offer as that instead of using dict.
            human_offer = self.utility_space_controller.convert_offer_dict_to_list(
                human_offer_dict
            )
            human_action.set_bid(human_offer)
            # Add user offer to the logger list.
            self.nego_history.add_to_history(
                human_action.get_bidder(),
                human_action.get_bid(),
                self.time_controller.get_current_time(),
                "",
                valance_arousal_predictions["Arousal"],
				valance_arousal_predictions["Valance"],
            )
            # Agent's generated offer for itself.
            (
                agent_action,
                agent_emotion,
                agent_emotion_file,
            ) = self.agent.receive_offer(human_action, valance_arousal_predictions)
            # Check if agent accepts the offer.
            if isinstance(agent_action, negoAction.Accept):
                if self.time_controller.time_is_up():
                    agent_num_of_emotions = self.agent.receive_end_of_deadline()
                    self.end_negotiation_with_no_result(agent_num_of_emotions)
                    break
                else:
                    # Send end of the negotiation action to the human as acceptance.
                    self.agent_interaction_controller.accept()
                    agent_num_of_emotions = self.agent.receive_accept(human_action)
                    self.end_negotiation_with_acceptance(agent_num_of_emotions)
                    break
            if self.time_controller.time_is_up():
                agent_num_of_emotions = self.agent.receive_end_of_deadline()
                self.end_negotiation_with_no_result(agent_num_of_emotions)
                break
            # Agent's offer to itself as list.
            agent_offer = agent_action.get_bid()
            # Agent's offer to user.
            agent_offer_to_human = (
                self.utility_space_controller.calculate_offer_for_opponent(agent_offer)
            )
            # Add agent offer to the logger list.
            self.nego_history.add_to_history(
                agent_action.get_bidder(),
                agent_offer,
                self.time_controller.get_current_time(),
                0,
                0,
                agent_emotion,
            )
            # Start the recording by calling api of the camera.
            self.camera_controller.start_recording()
            # Send offer to the human.
            self.send_agent_offer_to_human(agent_offer_to_human, agent_emotion_file)
            # Set negotiation gui according to agent's offer.
            self.set_negotiation_gui(agent_offer_to_human)
