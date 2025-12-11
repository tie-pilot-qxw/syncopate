import abc

class BaseHandler(abc.ABC):
    """
    Base class for all concrete handlers.
    """
    priority = 99
    @abc.abstractmethod
    def can_handle(self, data) -> bool:
        """
        Determine whether this handler can process the given data.
        Must be implemented by subclasses.

        Returns:
            bool: True if the handler can process the data, otherwise False.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def process(self, data):
        """
        Perform the handler-specific processing.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
# --- Subclass 1: handles strings only ---
class StringHandler(BaseHandler):
    priority = 10
    def can_handle(self, data) -> bool:
        # Match when the data is a string
        return isinstance(data, str)

    def process(self, data):
        print(f"[StringHandler]: processing string, uppercased -> {data.upper()}")

# --- Subclass 2: handles integers > 100 ---
class LargeIntHandler(BaseHandler):
    priority = 20
    def can_handle(self, data) -> bool:
        # Match when the data is an int greater than 100
        return isinstance(data, int) and data > 100

    def process(self, data):
        print(f"[LargeIntHandler]: processing large int, squared -> {data ** 2}")

# --- Subclass 3: handles non-empty lists ---
class ListHandler(BaseHandler):
    priority = 1
    def can_handle(self, data) -> bool:
        # Match when the data is a non-empty list
        return isinstance(data, list) and len(data) > 0

    def process(self, data):
        print(f"[ListHandler]: processing list, length -> {len(data)}")

# --- Subclass 4: handles smaller ints, demonstrates ordering ---
class SmallIntHandler(BaseHandler):
    priority = 5
    def can_handle(self, data) -> bool:
        return isinstance(data, int) and data <= 100
    
    def process(self, data):
        print(f"[SmallIntHandler]: processing small int, doubled -> {data * 2}")

def find_all_subclasses(cls):
    """Recursively find all subclasses of a class."""
    all_subclasses = [subclass for subclass in cls.__subclasses__()]
    # sort by priority
    sorted_subclasses = sorted(all_subclasses, key=lambda x: x.priority)
    for subclass in sorted_subclasses:
        print(f"Discovered subclass: {subclass.__name__} (priority: {subclass.priority})")
    
    return sorted_subclasses

def dispatch(data):
    """
    Automatically find and execute the appropriate handler.
    """
    print(f"\n--- Searching for handler for '{data}' (type: {type(data).__name__}) ---")
    
    # 1. Discover all subclasses of BaseHandler
    handler_classes = find_all_subclasses(BaseHandler)

    for cls in handler_classes:
        print(f"  Found handler class: {cls.__name__} (priority: {cls.priority})")

    # 2. Iterate through each handler in order
    for handler_class in handler_classes:
        # Instantiate handler
        handler_instance = handler_class()
        
        # 3. Call can_handle to test for a match
        print(f"    -> Trying {handler_class.__name__}...")
        if handler_instance.can_handle(data):
            print(f"    ✅ Matched! Processing with {handler_class.__name__}.")
            # 4. On match, process and stop searching
            handler_instance.process(data)
            return  # Exit after the first match

    # No handler found
    print("    ❌ No handler available for this data.")


# --- Quick demo ---
dispatch("hello world")
dispatch(200)
dispatch(50)
dispatch([1, 2, 3])
dispatch({"a": 1}) # No handler will match this type
