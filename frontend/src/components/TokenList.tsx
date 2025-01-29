import {
  Box,
  TableContainer,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
} from '@chakra-ui/react'
import { useState } from 'react'

interface Token {
  symbol: string
  address: string
}

// Mock data for supported tokens - replace with actual contract data later
const MOCK_TOKENS: Token[] = [
  { symbol: 'USDC', address: '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' },
  { symbol: 'USDT', address: '0xdac17f958d2ee523a2206206994597c13d831ec7' },
  { symbol: 'DAI', address: '0x6b175474e89094c44da98b954eedeac495271d0f' },
]

const TokenList = () => {
  const [tokens] = useState<Token[]>(MOCK_TOKENS)

  return (
    <Box w="full" overflowX="auto" bg="white" p={4} borderRadius="lg" borderWidth={1}>
      <Text fontSize="xl" mb={4}>Supported Tokens</Text>
      <TableContainer>
        <Table variant="simple">
          <Thead>
            <Tr>
              <Th>Token</Th>
              <Th>Address</Th>
            </Tr>
          </Thead>
          <Tbody>
            {tokens.map((token) => (
              <Tr key={token.address}>
                <Td>{token.symbol}</Td>
                <Td>
                  <Text noOfLines={1} maxW="300px">
                    {token.address}
                  </Text>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  )
}

export default TokenList 