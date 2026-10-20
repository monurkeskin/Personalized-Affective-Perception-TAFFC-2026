from xml.dom.minidom import parse
import xml.dom.minidom
import itertools


class UtilitySpaceController:
    def __init__(self, domain_file):
        # Role of the agent.
        self.domain_file = domain_file
        # Name to index dict.
        self.issue_name_to_index = {}
        # something.
        self.issue_index_to_name = {}
        # Call the role weight function for complete the weight dictionary.
        self.__set_domain_attributes()

    # Read the xml file of weights of keywords, then add to dictionary for calculation purpose.
    def __set_domain_attributes(self):
        # Open XML document using minidom parser
        DOMTree = xml.dom.minidom.parse(self.domain_file)
        collection = DOMTree.documentElement
        # Get utility space attributes.
        utility_space_obj = collection.getElementsByTagName("utility_space")[0]
        self.domain_type = utility_space_obj.attributes["domain_type"].value
        self.number_of_issues = int(
            utility_space_obj.attributes["number_of_issues"].value
        )

        if self.domain_type == "Resource Allocation":
            # Issue name and max count of it.
            self.issue_max_counts = []
            # Get issue list from the preference profile.
            issue_list = collection.getElementsByTagName("issue")
            # Iterate every issue and get max count of them.
            for issue in issue_list:
                issue_name = issue.attributes["name"].value
                issue_index = issue.attributes["index"].value
                issue_max_count = issue.attributes["max_count"].value
                self.issue_name_to_index[issue_name] = int(issue_index) - 1
                self.issue_index_to_name[int(issue_index) - 1] = issue_name
                self.issue_max_counts.append(int(issue_max_count))

    # Gets the offer and calculates leftover items for opponent in resource allocation type domains.
    def calculate_offer_for_opponent(self, offer):
        opponent_offer = []
        for issue_index, offered_amount in enumerate(offer):
            # Get issue name.
            issue_name = self.issue_index_to_name[issue_index]
            # Get max count of the issue.
            issue_max_count = self.issue_max_counts[issue_index]
            # Calculate leftover for the opponent of iterating issue.
            left_count = issue_max_count - int(offered_amount)
            opponent_offer.append(left_count)
        return opponent_offer

    def convert_offer_dict_to_list(self, offer_dict):
        offer_list = [None] * self.number_of_issues
        for (issue_name, issue_amount) in offer_dict.iteritems():
            # Get issue index.
            issue_index = self.issue_name_to_index[issue_name]
            # Append to the offer list with amount.
            offer_list[issue_index] = int(issue_amount)
        return offer_list

    def convert_offer_list_to_dict(self, offer_list):
        offer_dict = {}
        for issue_index, issue_amount in enumerate(offer_list):
            # Get issue name.
            issue_name = self.issue_index_to_name[issue_index]
            # Append to the offer dict with amount.
            offer_dict[issue_name] = int(issue_amount)
        return offer_dict

    def get_domain_type(self):
        return self.domain_type
