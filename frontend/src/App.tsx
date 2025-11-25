import { ChakraProvider } from '@chakra-ui/react'
import { Flex }  from "@chakra-ui/react"
import Sidebar from "./components/Sidebar";
import Chatbot from "./components/Chatbot";
import "./i18n";

function App() {
  return (
    <ChakraProvider>
      <Flex direction="row" align="stretch" h="100vh" w="100vw">
        <Sidebar/>
        <Chatbot
          endpoint="http://localhost:8000/search"
        />
      </Flex>
    </ChakraProvider>
  )
}

export default App;
