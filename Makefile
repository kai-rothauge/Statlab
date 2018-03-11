include $(HOME)/Software/Elemental/conf/ElVars

SRC_PATH = $(HOME)/Projects/Statlab
TARGET_PATH = $(HOME)/Projects/Statlab

# put libEl's CXXFLAGS in front so ours can override it
CXXFLAGS += $(EL_COMPILE_FLAGS) -fdiagnostics-color=always
#CXXFLAGS += -Wall
CXXFLAGS += -Wno-unused -Wno-reorder -std=c++14 -fext-numeric-literals -fopenmp

LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)

OBJ_FILES = \
	$(TARGET_PATH)/Statlab.o \
	$(TARGET_PATH)/leverage.o \
	$(TARGET_PATH)/delete1.o \

$(TARGET_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) -c $(CXXFLAGS) -D_GLIBCXX_USE_CXX11_ABI=1 $< -o $@

.PHONY: default
default: $(TARGET_PATH)/test

$(TARGET_PATH)/test: $(TARGET_PATH) $(OBJ_FILES)
	$(CXX) -dynamic $(CXXFLAGS) -D_GLIBCXX_USE_CXX11_ABI=1 -o $@ $(OBJ_FILES) $(LDFLAGS)

$(TARGET_PATH):
	mkdir -p $(TARGET_PATH)

.PHONY: clean
clean:
	rm -rf $(TARGET_PATH)
