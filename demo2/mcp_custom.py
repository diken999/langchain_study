from fastmcp import FastMCP

# 初始化mcp
mcp = FastMCP("Math")

@mcp.tool()
def add(x: float, y: float) -> float:
  """两数相加"""
  return x + y

@mcp.tool()
def multiply(x: float, y: float) -> float:
  """两数相乘"""
  return x * y

# 启动mcp服务，通信方式设置为stdio
if __name__ == "__main__":
  mcp.run(transport="stdio")