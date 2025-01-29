import { Button, Text, VStack, useToast } from '@chakra-ui/react'
import { useAccount, useConnect, useDisconnect } from 'wagmi'
import { InjectedConnector } from '@wagmi/core/connectors/injected'

const ConnectWallet = () => {
  const { address, isConnected } = useAccount()
  const { connect } = useConnect()
  const { disconnect } = useDisconnect()
  const toast = useToast()

  const handleConnect = async () => {
    try {
      await connect({ connector: new InjectedConnector() })
      toast({
        title: 'Connected',
        description: 'Wallet connected successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to connect wallet',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
    }
  }

  return (
    <VStack spacing={4}>
      {isConnected ? (
        <>
          <Text>Connected to {address}</Text>
          <Button colorScheme="red" onClick={() => disconnect()}>
            Disconnect
          </Button>
        </>
      ) : (
        <Button colorScheme="blue" onClick={handleConnect}>
          Connect Wallet
        </Button>
      )}
    </VStack>
  )
}

export default ConnectWallet 