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

#import nothing

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

    def __init__(self):
        self.data = [
            ["first_dummy_object",0],
            ["second_dummy_object",None]
        ]

        self.data = [
            #"dummy_object": id(self.data["dummy_object"])
            # "first_dummy_object": id(self.data["second_dummy_object"]),
            # "second_dummy_object":None
            ["first_dummy_object",id(self.data[-1][1])],
            ["second_dummy_object", None]
        ]
        print("I have been created")


    def add(self, object):
        # Adds a node at the end
        self.data[-1][1] = id(object)
        self.data.append([object, None])

    def add_first(self, object):
        new_head = [object, id(self.data[0][1])]
        self.data.insert(0, new_head)

    def remove_head(self):
        del self.data[0]



test = linked_list()
print(test.data)
test.add(None)
print("="*25)
print(test.data)
print("="*25)

# Object ids get changed after every change because "immutable"
#print(f"I sure hope this gets put as first pointer: {id(poggers.data[0][0])}")
test.add_first(None)
print(test.data)
print("Removing Head")
test.remove_head()
print(test.data)
test.add_first(None)
print(test.data)
test.add(None)
test.add("Hello")
print(test.data)
# object = "hello"
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

