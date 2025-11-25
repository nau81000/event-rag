
import React from "react";
import { Flex, Text, Box } from "@chakra-ui/react";
import backgroundImage from "../assets/events.png";
import { useTranslation } from "react-i18next";

const Sidebar: React.FC = (
{
}) => {
  const { t } = useTranslation();
  return (
    <Flex
      width="500px"
      height="100vh"
      direction="column"
      bg="gray.700"
      color="white"
    >
      <Box
        height="95%"
        backgroundImage={`url(${backgroundImage})`}
        backgroundSize="cover"
        backgroundPosition="center"
        backgroundRepeat="no-repeat"
        filter="brightness(0.5)"
      />

      <Flex justify="center">
        <Text mt={4} opacity={0.3}>
          {t("poweredBy")}
        </Text>
      </Flex>
    </Flex>
  );
};

export default Sidebar
