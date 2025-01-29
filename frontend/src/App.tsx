import { ChakraProvider, Container, VStack, Heading } from '@chakra-ui/react'
import { WagmiConfig, createConfig } from 'wagmi'
import { baseGoerli } from 'wagmi/chains'
import { createPublicClient, http } from 'viem'
import ConnectWallet from './components/ConnectWallet'
import ArbitragePanel from './components/ArbitragePanel'
import TokenList from './components/TokenList'

const config = createConfig({
  publicClient: createPublicClient({
    chain: baseGoerli,
    transport: http()
  })
})

function App() {
  return (
    <WagmiConfig config={config}>
      <ChakraProvider>
        <Container maxW="container.lg" py={8}>
          <VStack align="stretch" spacing={8}>
            <Heading textAlign="center">Flash Loan Arbitrage Dashboard</Heading>
            <ConnectWallet />
            <ArbitragePanel />
            <TokenList />
          </VStack>
        </Container>
      </ChakraProvider>
    </WagmiConfig>
  )
}

export default App
