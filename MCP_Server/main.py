"""
FastMCP server with a toy database and various tools.
Run with: uv run mcp install main.py
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Optional
from datetime import datetime
import json

# Create an MCP server
mcp = FastMCP("ToyDatabaseServer")

# Toy database - in-memory data structures
users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 28, "city": "New York"},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 32, "city": "San Francisco"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 25, "city": "Chicago"},
    {"id": 4, "name": "Diana", "email": "diana@example.com", "age": 29, "city": "New York"},
]

products_db = [
    {"id": 101, "name": "Laptop", "price": 999.99, "category": "Electronics", "stock": 15},
    {"id": 102, "name": "Coffee Mug", "price": 12.50, "category": "Kitchen", "stock": 100},
    {"id": 103, "name": "Headphones", "price": 79.99, "category": "Electronics", "stock": 25},
    {"id": 104, "name": "Notebook", "price": 8.99, "category": "Office", "stock": 50},
]

orders_db = [
    {"id": 1001, "user_id": 1, "product_id": 101, "quantity": 1, "date": "2024-01-15"},
    {"id": 1002, "user_id": 2, "product_id": 102, "quantity": 2, "date": "2024-01-16"},
    {"id": 1003, "user_id": 1, "product_id": 103, "quantity": 1, "date": "2024-01-17"},
]

# User management tools
@mcp.tool()
def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user details by user ID"""
    for user in users_db:
        if user["id"] == user_id:
            return user
    return None

@mcp.tool()
def get_users_by_city(city: str) -> List[Dict]:
    """Get all users from a specific city"""
    return [user for user in users_db if user["city"].lower() == city.lower()]

@mcp.tool()
def create_user(name: str, email: str, age: int, city: str) -> Dict:
    """Create a new user in the database"""
    new_id = max(user["id"] for user in users_db) + 1
    new_user = {
        "id": new_id,
        "name": name,
        "email": email,
        "age": age,
        "city": city
    }
    users_db.append(new_user)
    return new_user

# Product management tools
@mcp.tool()
def get_product_by_id(product_id: int) -> Optional[Dict]:
    """Get product details by product ID"""
    for product in products_db:
        if product["id"] == product_id:
            return product
    return None

@mcp.tool()
def get_products_by_category(category: str) -> List[Dict]:
    """Get all products in a specific category"""
    return [product for product in products_db 
            if product["category"].lower() == category.lower()]

@mcp.tool()
def update_product_stock(product_id: int, new_stock: int) -> Optional[Dict]:
    """Update the stock quantity of a product"""
    for product in products_db:
        if product["id"] == product_id:
            product["stock"] = new_stock
            return product
    return None

# Order management tools
@mcp.tool()
def get_user_orders(user_id: int) -> List[Dict]:
    """Get all orders for a specific user"""
    user_orders = []
    for order in orders_db:
        if order["user_id"] == user_id:
            # Enrich order with user and product details
            user = get_user_by_id(user_id)
            product = get_product_by_id(order["product_id"])
            enriched_order = order.copy()
            enriched_order["user_name"] = user["name"] if user else "Unknown"
            enriched_order["product_name"] = product["name"] if product else "Unknown"
            user_orders.append(enriched_order)
    return user_orders

@mcp.tool()
def create_order(user_id: int, product_id: int, quantity: int) -> Optional[Dict]:
    """Create a new order"""
    # Check if user exists
    user = get_user_by_id(user_id)
    if not user:
        return None
    
    # Check if product exists and has enough stock
    product = get_product_by_id(product_id)
    if not product or product["stock"] < quantity:
        return None
    
    # Create order
    new_id = max(order["id"] for order in orders_db) + 1 if orders_db else 1001
    new_order = {
        "id": new_id,
        "user_id": user_id,
        "product_id": product_id,
        "quantity": quantity,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    orders_db.append(new_order)
    
    # Update product stock
    update_product_stock(product_id, product["stock"] - quantity)
    
    # Enrich order with details
    enriched_order = new_order.copy()
    enriched_order["user_name"] = user["name"]
    enriched_order["product_name"] = product["name"]
    enriched_order["total_price"] = product["price"] * quantity
    
    return enriched_order

# Analytics tools
@mcp.tool()
def get_sales_by_category() -> Dict:
    """Get total sales amount by product category"""
    sales = {}
    for order in orders_db:
        product = get_product_by_id(order["product_id"])
        if product:
            category = product["category"]
            amount = product["price"] * order["quantity"]
            sales[category] = sales.get(category, 0) + amount
    return sales

@mcp.tool()
def get_user_statistics() -> Dict:
    """Get statistics about users"""
    total_users = len(users_db)
    avg_age = sum(user["age"] for user in users_db) / total_users if total_users > 0 else 0
    cities = {}
    
    for user in users_db:
        cities[user["city"]] = cities.get(user["city"], 0) + 1
    
    return {
        "total_users": total_users,
        "average_age": round(avg_age, 2),
        "users_by_city": cities
    }

# Resource for user data
@mcp.resource("user://{user_id}")
def get_user_resource(user_id: str) -> str:
    """Get user data as a resource"""
    user = get_user_by_id(int(user_id))
    if user:
        return json.dumps(user, indent=2)
    return f"User with ID {user_id} not found"

# Resource for product catalog
@mcp.resource("catalog://{category}")
def get_catalog_resource(category: str) -> str:
    """Get product catalog by category"""
    products = get_products_by_category(category)
    return json.dumps(products, indent=2)

# Prompt templates
@mcp.prompt()
def generate_user_report(user_id: int) -> str:
    """Generate a comprehensive report for a user"""
    return f"""
Please generate a comprehensive report for user ID {user_id}. 
Include:
1. User details and contact information
2. All orders placed by the user with product details
3. Total spending amount
4. Purchase history analysis
5. Personalized recommendations based on their purchase history

Format the report in a clear, professional manner.
"""

@mcp.prompt()
def generate_sales_summary() -> str:
    """Generate a sales summary report"""
    return """
Please generate a sales summary report including:
1. Total revenue by product category
2. Best selling products
3. User demographics analysis
4. Sales trends and insights
5. Recommendations for inventory management and marketing

Present the data with clear metrics and actionable insights.
"""

# Utility tools
@mcp.tool()
def search_users(query: str) -> List[Dict]:
    """Search users by name or email"""
    query = query.lower()
    results = []
    for user in users_db:
        if (query in user["name"].lower() or 
            query in user["email"].lower() or 
            query in user["city"].lower()):
            results.append(user)
    return results

@mcp.tool()
def get_low_stock_products(threshold: int = 10) -> List[Dict]:
    """Get products with low stock (below threshold)"""
    return [product for product in products_db if product["stock"] < threshold]

if __name__ == "__main__":
    # Initialize with some sample data if needed
    print("Toy Database MCP Server started!")
    print("Available tools:")
    print("- User management: get_user_by_id, get_users_by_city, create_user")
    print("- Product management: get_product_by_id, get_products_by_category, update_product_stock")
    print("- Order management: get_user_orders, create_order")
    print("- Analytics: get_sales_by_category, get_user_statistics")
    print("- Utilities: search_users, get_low_stock_products")