import { Menu, MenuButton, MenuList, MenuItem, IconButton, Flex, Text } from "@chakra-ui/react";
import { useTranslation } from "react-i18next";
import { FiGlobe } from "react-icons/fi";

export function ChatbotHeader() {
  const { i18n, t } = useTranslation();

  const setLang = (lng: "en" | "fr") => i18n.changeLanguage(lng);
  const current = i18n.language.startsWith("fr") ? "fr" : "en";

  return (
    <Flex w="100%" h="60px" px={4} borderBottomWidth="1px" bg="transparent">
        <Flex w="95%" align="center" justify="center" color="white">
            <Text fontSize="3xl" fontWeight="bold">
                {t("appTitle")}
            </Text>
        </Flex>
        <Flex align="center" justify="right">
          <Menu placement="bottom-end">
          <MenuButton
            as={IconButton}
            aria-label="Language selector"
            icon={<FiGlobe size={20} />}
            variant="ghost"
            bg="transparent"
            color="white"
            _hover={{ bg: "gray.500" }}
            _active={{ bg: "gray.500" }}
            _focus={{ boxShadow: "none" }}
          />
          <MenuList minW="140px" bg="gray.500" borderColor="gray.600">
            <MenuItem
              bg="gray.500"
              _hover={{ bg: "gray.300" }}
              onClick={() => setLang("en")}
              fontWeight={current === "en" ? "bold" : "normal"}
            >
              English
            </MenuItem>
            <MenuItem
              bg="gray.500"
              _hover={{ bg: "gray.300" }}
              onClick={() => setLang("fr")}
              fontWeight={current === "fr" ? "bold" : "normal"}
            >
              Français
            </MenuItem>
          </MenuList>
        </Menu>        
        </Flex>
    </Flex>
  );
}