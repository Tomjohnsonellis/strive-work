"""
Bonus Task #1: Linked List

Build a complete functional Linked_List class
(with all checks and balances so it doesn't fail with Nones)

Methods Required:
        add(node): add a node in the end of the list
        add_first(node): adds a node to the beginning of the list
        add_at(node, index): add a node in the given position
        remove(E): removes the first node that contains that element
        remove_all(E): remove every node that contains that element
        remove_head(): removes the first element of the list

    Tostring: a native python method that makes it so when the list is printed it prints the
    entirety of the list. so if my list is: 1->3->4 it would print 1,3,4. Research is suggested.

-= NO IMPORTING ALLOWED =-

A linked list is a structure based on nodes, where every node has 2 attributes:
A stored element ("Apple")
A pointer to the next node (memory location)
"""


# import nothing

## First attempt was using dictionaries, then decided against it
# class linked_list:
#
#     def __init__(self):
#         self.data = {
#             "first_dummy_object":0,
#             "second_dummy_object":None
#         }
#
#         self.data = {
#             #"dummy_object": id(self.data["dummy_object"])
#             "first_dummy_object": id(self.data["second_dummy_object"]),
#             "second_dummy_object":None
#         }
#         print("I exist")
#
#
#     def add(self, object):
#         self.append(object)


class linked_list:

    def __init__(self, list_of_elements=None):
        if type(list_of_elements) == list:

            self.data = [[list_of_elements[0], None]]
            if len(list_of_elements) >= 2:
                for element in list_of_elements[1:]:
                    self.add(element)
        else:
            self.data = [
                ["first_dummy_object", 0],
                ["second_dummy_object", None]
            ]

            self.data = [
                ["first_dummy_object", id(self.data[-1][1])],
                ["second_dummy_object", None]
            ]

    def add(self, object=None):
        # Adds a node at the end
        if object:
            self.data.append([object, None])
            self.data[-2][1] = id(object)

    def add_first(self, object=None):
        if object:
            # We need a pointer to the original head's data
            new_head = [object, id(self.data[0][0])]
            self.data.insert(0, new_head)

    def remove_head(self):
        if len(self.data) > 1:
            del self.data[0]
        else:
            print("That will delete me entirely!")

    def add_at(self, object=None, index=0):
        if object:
            if index == 0:
                self.add_first(object)
            else:
                # Split the data
                before_data = self.data[:index]
                after_data = self.data[index:]
                # Update the pointer that we are affecting
                before_data[-1][1] = id(object)
                # Create a node to add, [data, pointer to next data]
                node_to_insert = [object, id(after_data[0][0])]
                # Put them all together
                rebuilt_linked_list = before_data
                rebuilt_linked_list.append(node_to_insert)
                for node in after_data:
                    rebuilt_linked_list.append(node)
                self.data = rebuilt_linked_list

    def remove(self, element=None):
        if element:
            for index, node in enumerate(self.data):
                if node[0] == element and node[1] is not None:
                    print("Match Found")
                    # Split the data
                    before_data = self.data[:index]
                    after_data = self.data[index + 1:]
                    # Update pointer before removed node
                    before_data[-1][1] = id(after_data[0][0])
                    # Rebuild
                    rebuilt_linked_list = before_data
                    for block in after_data:
                        rebuilt_linked_list.append(block)
                    self.data = rebuilt_linked_list
                    # Return back as we are only removing one element
                    return
                if node[0] == element and node[1] is None:
                    del self.data[-1]
                    self.data[-1][1] = None
                    # Return back as we are only removing one element
                    return

    def remove_all(self, element=None):
        if element:
            # This is a lazy solution, but seems to work fine
            for item in range(len(self.data)):
                try:
                    self.remove(element)
                except:
                    break

    def tostring(self):
        stringlist = []
        for node in self.data:
            stringlist.append(node[0])
        #print(str(stringlist))
        return str(stringlist)


def test_method(command):
    # This would be nice so that I just iterate through a list of commands like [.add(5), .remove_head()]
    # No idea how to do that at the moment
    print(f"I will perform: {str(command)}")
    print(">>>>>Resulting in:")
    print("results")


def show(linked_list):
    print(">>>>>Resulting in:")
    print(linked_list.data)
    return


def test_all(test_linked_list):
    print("#" * 25)
    print("#" * 25)
    print("START")
    print("#" * 25)
    print("#" * 25)
    # test_linked_list = linked_list(["I am a list of one element"])
    print(test_linked_list.data)
    print(">>>>>I will now use .add(\"Added\")")
    test_linked_list.add("Added")
    show(test_linked_list)
    print(">>>>>I will now use .add_first(\"First\")")
    test_linked_list.add_first("First")
    show(test_linked_list)
    print(">>>>>I will now use .remove_head()")
    test_linked_list.remove_head()
    show(test_linked_list)
    print(">>>>>I will now use .add(\"Egg\")")
    test_linked_list.add("Egg")
    show(test_linked_list)
    print(">>>>>I will now use .add(\"Spice\")")
    test_linked_list.add("Spice")
    show(test_linked_list)
    print(">>>>>I will now use .add(1080)")
    test_linked_list.add(1080)
    show(test_linked_list)
    print(">>>>>I will now use add_at('#INSERTED#', 3)")
    test_linked_list.add_at("#INSERTED#", 3)
    show(test_linked_list)
    print(">>>>>I will now use .add(\"Dupes\")")
    test_linked_list.add("Dupes")
    show(test_linked_list)
    print(">>>>>I will now use .add(\"Dupes\")")
    test_linked_list.add("Dupes")
    show(test_linked_list)
    print(">>>>>I will now use .add(\"Dupes\")")
    test_linked_list.add("Dupes")
    show(test_linked_list)
    print("######################### REMOVALS")
    print(">>>>>I will now use remove(\"Dupes\")")
    test_linked_list.remove("Dupes")
    show(test_linked_list)
    print(">>>>>I will now use remove_all(\"Dupes\")")
    test_linked_list.remove_all("Dupes")
    show(test_linked_list)
    print("######################### To String")
    final_test = test_linked_list.tostring()
    print(final_test)
    print(type(final_test))


if __name__ == '__main__':
    dummy_list = linked_list(["I am a list of one element"])
    test_all(dummy_list)

## Abandoned first draft
# # object = "hello"
#
# print(id(object))
#
# object_list = [object]
# pointer_list = [id(object)]
#
# print(id(object_list))
# print(id(object_list[0]))
#
# nodes = zip(object_list, pointer_list)
# print(nodes)
# for object, pointer in nodes:
#     print(f"{object} @ {pointer}")
