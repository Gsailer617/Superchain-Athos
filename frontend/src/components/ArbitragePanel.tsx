import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  useToast,
  UseToastOptions,
} from '@chakra-ui/react'
import { useState } from 'react'
import { useAccount } from 'wagmi'

const ArbitragePanel = () => {
  const { isConnected } = useAccount()
  const [amount, setAmount] = useState('')
  const toast = useToast()

  const showToast = (options: UseToastOptions) => {
    toast({
      duration: 3000,
      isClosable: true,
      ...options,
    })
  }

  const handleArbitrage = async () => {
    if (!isConnected) {
      showToast({
        title: 'Error',
        description: 'Please connect your wallet first',
        status: 'error',
      })
      return
    }

    try {
      // TODO: Implement arbitrage logic here
      showToast({
        title: 'Success',
        description: 'Arbitrage operation initiated',
        status: 'success',
      })
    } catch (error) {
      showToast({
        title: 'Error',
        description: 'Failed to execute arbitrage',
        status: 'error',
      })
    }
  }

  return (
    <Box w="full" maxW="md" p={6} borderWidth={1} borderRadius="lg" bg="white">
      <VStack spacing={4}>
        <FormControl>
          <FormLabel>Flash Loan Amount</FormLabel>
          <Input
            type="number"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="Enter amount"
          />
        </FormControl>
        <Button
          colorScheme="green"
          onClick={handleArbitrage}
          disabled={!isConnected || !amount}
          width="full"
        >
          Execute Arbitrage
        </Button>
      </VStack>
    </Box>
  )
}

export default ArbitragePanel 