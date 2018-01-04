TOPDIR := ..

PLATFORM = common
include $(TOPDIR)/$(PLATFORM).mak

BUILD_FOLDER ?= .build
TARGET ?= mesh_merge
BIN_DIR ?= .

# Source files
OBJECTS += $(addprefix $(BUILD_FOLDER)/, \
        mesh_merge.o)

all: $(BIN_DIR)/$(TARGET)

$(BUILD_FOLDER):
	$(PSILENT)mkdir -p $@

$(OBJECTS): $(TOPDIR)/$(PLATFORM).mak | $(BUILD_FOLDER)

-include $(OBJECTS:%.o=%.d)

C_INC += -I$(TOPDIR)

$(BUILD_FOLDER)/%.o: %.cpp
	@echo "..Building $< ..."
	$(PSILENT)$(CCACHE_EXT) $(CXX) -c -MMD $(CXX_FLAGS) $(C_INC) $< -o $@
	@echo "..Done ($<)"

$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "..Linking $@ ..."
	$(PSILENT)$(CCACHE_EXT) $(LD) $(OBJECTS) $(L_FLAGS) -o $@
	@echo "..Done ($@)"

clean: $(BIN_DIR)/$(TARGET)_clean

$(BIN_DIR)/$(TARGET)_clean:
	@echo "..Cleaning $(BIN_DIR)/$(TARGET) ..."
	$(PSILENT)rm -f $(BIN_DIR)/$(TARGET)
	$(PSILENT)rm -rf $(BUILD_FOLDER)
	@echo "..Done"

help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (default), clean, mesh_merge"

.PHONY: all clean $(BIN_DIR)/$(TARGET)_clean help

