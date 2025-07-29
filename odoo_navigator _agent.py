#!/usr/bin/env python3
"""
Odoo Navigation AI Agent - Improved Version
Specialized AI agent for navigating Odoo pages using LangGraph and Groq API
"""

import os
import sys
import argparse
import getpass
import re
import json
import xmlrpc.client
from typing import Dict, Any, List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ====== CONFIGURATION ======
def parse_arguments():
    parser = argparse.ArgumentParser(description='Odoo Navigation AI Agent')
    parser.add_argument('--host', default='localhost', help='Odoo host (default: localhost)')
    parser.add_argument('--port', type=int, default=8069, help='Odoo port (default: 8069)')
    parser.add_argument('--db', default='odoo', help='Odoo database name (default: odoo)')
    parser.add_argument('--user', default='mayssenbhamor@gmail.com', help='Odoo username')
    parser.add_argument('--password', help='Odoo password (will prompt if not provided)')
    parser.add_argument('--command', help='Run a single command and exit')
    parser.add_argument('--groq-api-key', help='Groq API key (will use GROQ_API_KEY env var if not provided)')
    
    return parser.parse_args()

# ====== STATE DEFINITION ======
class AgentState(TypedDict):
    user_input: str
    parsed_intent: Dict[str, Any]
    odoo_url: str
    status: str
    error_message: str
    response: str

# ====== IMPROVED ODOO CONNECTOR ======
class OdooConnector:
    def __init__(self, host: str, port: int, db: str, username: str, password: str):
        """Initialize connection to Odoo"""
        self.host = host
        self.port = port
        self.db = db
        self.username = username
        self.password = password
        
        # XML-RPC endpoints
        self.common_url = f'http://{host}:{port}/xmlrpc/2/common'
        self.object_url = f'http://{host}:{port}/xmlrpc/2/object'
        
        # Authenticate and get user ID
        self.common = xmlrpc.client.ServerProxy(self.common_url)
        self.uid = self.common.authenticate(db, username, password, {})
        
        if not self.uid:
            raise Exception("❌ Authentication failed. Check your DB, username, and password.")
        
        # Create object endpoint proxy
        self.models = xmlrpc.client.ServerProxy(self.object_url)
        
        # Get current company ID for cids parameter
        self.company_id = self._get_current_company_id()
    
    def _get_current_company_id(self) -> int:
        """Get the current user's company ID"""
        try:
            user_data = self.models.execute_kw(
                self.db, self.uid, self.password,
                'res.users', 'read',
                [self.uid], {'fields': ['company_id']}
            )
            if user_data and user_data[0].get('company_id'):
                return user_data[0]['company_id'][0]
            return 1  # Default fallback
        except:
            return 1  # Default fallback
    
    def search_read(self, model: str, domain: List[Tuple], fields: List[str] = None, 
                   limit: int = None, offset: int = 0, order: str = None) -> List[Dict[str, Any]]:
        """Search and read records from a model"""
        if fields is None:
            fields = ['name', 'id']
        
        kwargs = {
            'fields': fields,
            'offset': offset
        }
        
        if limit:
            kwargs['limit'] = limit
            
        if order:
            kwargs['order'] = order
        
        return self.models.execute_kw(
            self.db, self.uid, self.password,
            model, 'search_read',
            [domain], kwargs
        )
    
    def get_menu_with_action_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get menu with detailed action information"""
        # Search for menu items
        menus = self.search_read(
            'ir.ui.menu',
            [('name', 'ilike', name)],
            ['id', 'name', 'action', 'parent_id'],
            limit=5  # Get multiple matches to find the best one
        )
        
        if not menus:
            return None
        
        # Find the best matching menu (prefer exact matches or main menus)
        best_menu = None
        for menu in menus:
            if menu['name'].lower() == name.lower():
                best_menu = menu
                break
            elif not best_menu:  # Take the first one as fallback
                best_menu = menu
        
        if not best_menu or not best_menu.get('action'):
            return None
        
        # Extract action details
        action_ref = best_menu['action']
        if isinstance(action_ref, list) and len(action_ref) >= 2:
            action_model = action_ref[0]
            action_id = action_ref[1]
        elif isinstance(action_ref, str) and ',' in action_ref:
            parts = action_ref.split(',')
            action_model = parts[0]
            action_id = int(parts[1]) if len(parts) > 1 else None
        else:
            return best_menu  # Return basic menu info
        
        # Get detailed action information
        try:
            action_details = None
            if action_model == 'ir.actions.act_window':
                action_details = self.search_read(
                    'ir.actions.act_window',
                    [('id', '=', action_id)],
                    ['id', 'name', 'res_model', 'view_mode', 'domain', 'context'],
                    limit=1
                )
            elif action_model == 'ir.actions.client':
                action_details = self.search_read(
                    'ir.actions.client',
                    [('id', '=', action_id)],
                    ['id', 'name', 'tag', 'context', 'params'],
                    limit=1
                )
            
            if action_details:
                best_menu['action_details'] = action_details[0]
                best_menu['action_id'] = action_id
                best_menu['action_model'] = action_model
            
        except Exception as e:
            print(f"Warning: Could not fetch action details: {e}")
        
        return best_menu
    
    def get_action_by_model(self, model: str) -> Optional[Dict[str, Any]]:
        """Get action for a model with more details"""
        actions = self.search_read(
            'ir.actions.act_window',
            [('res_model', '=', model)],
            ['id', 'name', 'view_mode', 'domain', 'context', 'res_model'],
            limit=1
        )
        
        if actions:
            return actions[0]
        return None
    
    def get_base_url(self) -> str:
        """Get the base URL for web links"""
        return f"http://{self.host}:{self.port}"
    
    def get_company_id(self) -> int:
        """Get the company ID for URL construction"""
        return self.company_id

# ====== ENHANCED INTENT PARSER WITH GROQ ======
class EnhancedIntentParser:
    def __init__(self, groq_api_key: str):
        """Initialize with Groq API key"""
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=groq_api_key,
            temperature=0.1
        )
        
        # Enhanced navigation mapping
        self.navigation_mapping = {
            "sales": ["Sales", "sale"],
            "sale": ["Sales", "sale"],
            "invoices": ["Invoicing", "invoicing"],
            "invoicing": ["Invoicing", "invoicing"], 
            "customers": ["Contacts", "contacts"],
            "contacts": ["Contacts", "contacts"],
            "products": ["Products", "product"],
            "product": ["Products", "product"],
            "inventory": ["Inventory", "inventory"],
            "stock": ["Inventory", "inventory"],
            "purchases": ["Purchase", "purchase"],
            "purchase": ["Purchase", "purchase"],
            "employees": ["Employees", "employee"],
            "employee": ["Employees", "employee"],
            "settings": ["Settings", "settings"],
            "apps": ["Apps", "apps"],
            "modules": ["Apps", "apps"],
            "dashboard": ["Dashboards", "dashboard"],
            "dashboards": ["Dashboards", "dashboard"],
            "calendar": ["Calendar", "calendar"],
            "discuss": ["Discuss", "discuss"],
            "inbox": ["Inbox", "inbox"],
            "messages": ["Inbox", "inbox"],
            "crm": ["CRM", "crm"],
            "leads": ["Leads", "leads"],
            "opportunities": ["Opportunities", "opportunities"],
            "website": ["Website", "website"],
            "pos": ["Point of Sale", "pos"],
            "manufacturing": ["Manufacturing", "manufacturing"],
            "projects": ["Projects", "projects"],
        }
        
        # Enhanced module mapping
        self.module_mapping = {
            "sales": "sale.order",
            "sale orders": "sale.order",
            "sale order": "sale.order",
            "orders": "sale.order",
            "invoices": "account.move",
            "invoice": "account.move",
            "invoicing": "account.move",
            "bills": "account.move",
            "customers": "res.partner",
            "customer": "res.partner",
            "partners": "res.partner",
            "partner": "res.partner",
            "contacts": "res.partner",
            "products": "product.template",
            "product": "product.template",
            "inventory": "stock.quant",
            "stock": "stock.quant",
            "purchases": "purchase.order",
            "purchase orders": "purchase.order",
            "purchase": "purchase.order",
            "employees": "hr.employee",
            "employee": "hr.employee",
            "users": "res.users",
            "user": "res.users",
            "payments": "account.payment",
            "payment": "account.payment",
            "leads": "crm.lead",
            "lead": "crm.lead",
            "opportunities": "crm.lead",
            "opportunity": "crm.lead",
            "settings": "res.config.settings",
            "configuration": "res.config.settings",
            "apps": "ir.module.module",
            "modules": "ir.module.module",
            "dashboard": "board.board",
            "dashboards": "board.board",
            "inbox": "mail.message",
            "discuss": "mail.message",
            "messages": "mail.message",
        }
    
    def parse(self, user_input: str) -> Dict[str, Any]:
        """Parse user input using Groq AI and fallback to rule-based parsing"""
        user_input = user_input.lower().strip()
        
        if not user_input:
            return {
                "type": "unknown",
                "entity": "",
                "entity_model": None,
                "navigation_targets": [],
                "original_input": user_input
            }
        
        try:
            # Use Groq to understand the intent
            groq_result = self._parse_with_groq(user_input)
            
            # Validate and enhance with rule-based mapping
            if groq_result and groq_result.get("entity"):
                entity = groq_result["entity"].lower()
                navigation_targets = self._map_to_navigation_targets(entity)
                entity_model = self._map_to_odoo_model(entity)
                
                if navigation_targets or entity_model:
                    return {
                        "type": "navigation",
                        "entity": entity,
                        "entity_model": entity_model,
                        "navigation_targets": navigation_targets,
                        "original_input": user_input
                    }
            
            # Fallback to rule-based parsing
            return self._fallback_parse(user_input)
            
        except Exception as e:
            print(f"Error with Groq parsing: {e}")
            # Fallback to rule-based parsing
            return self._fallback_parse(user_input)
    
    def _parse_with_groq(self, user_input: str) -> Dict[str, Any]:
        """Use Groq to parse the user input"""
        system_prompt = """You are an expert at understanding user intents for Odoo ERP navigation with advanced reasoning capabilities.

    Your task is to extract the COMPLETE navigation path from user input, preserving the full context when users specify compound destinations.

    Odoo Navigation Hierarchy (from general to specific):
    1. Main Modules: sales, invoices, customers, contacts, products, inventory, purchases, employees, settings, apps
    2. Sub-modules: 
       - Sales: quotations, orders, analysis, products
       - Invoices: customer invoices, vendor bills, analysis
       - Products: items, variants, categories, analysis
       - Inventory: transfers, levels, valuation, reporting
       - CRM: leads, opportunities, pipeline, analysis
    3. Views: list, kanban, form, graph, pivot, calendar
    4. Reports: analysis, reporting, dashboard, statistics

    CRITICAL REASONING PROCESS:
    1. Identify ALL navigation components in the input (module, sub-module, view type, report type)
    2. Preserve the FULL PATH when compound destinations are specified
    3. For analysis/report requests, maintain the context (e.g., "product sales analysis" should include both "products" and "sales")
    4. When multiple entities are present, combine them in the order mentioned to form the complete path
    5. Only simplify to a single entity when the input clearly requests a top-level module

    IMPORTANT RULES:
    - "Analysis/Report" keywords indicate the user wants the analytical view of the specified path
    - Compound phrases like "sales product analysis" should be kept intact as they represent specific views
    - View types (list, kanban, etc.) should be included when specified
    - Maintain prepositional context (e.g., "analysis OF products" vs "products analysis")
    - For ambiguous cases, prefer the more complete path over simplifying

    Return a JSON object with:
    - "type": "navigation" if it's a navigation request, "unknown" otherwise  
    - "entity": the complete navigation path (e.g., "sales/products/analysis")
    - "all_entities": list of all components found in the input
    - "reasoning": brief explanation of your parsing decision
    - "confidence": a number from 0-1 indicating your confidence

    Examples with reasoning:
    - "go to sales" -> {"type": "navigation", "entity": "sales", "all_entities": ["sales"], "reasoning": "Single top-level module requested", "confidence": 0.95}
    - "sales product analysis" -> {"type": "navigation", "entity": "sales/products/analysis", "all_entities": ["sales", "products", "analysis"], "reasoning": "Full analytical view path requested", "confidence": 0.9}
    - "show me product sales in pivot view" -> {"type": "navigation", "entity": "products/sales/pivot", "all_entities": ["products", "sales", "pivot"], "reasoning": "Specific view type requested with context", "confidence": 0.85}
    - "customer invoices report" -> {"type": "navigation", "entity": "invoices/customers/report", "all_entities": ["invoices", "customers", "report"], "reasoning": "Report view with customer context", "confidence": 0.9}
    - "inventory valuation analysis" -> {"type": "navigation", "entity": "inventory/valuation/analysis", "all_entities": ["inventory", "valuation", "analysis"], "reasoning": "Specific analytical view requested", "confidence": 0.9}
    - "open products list" -> {"type": "navigation", "entity": "products/list", "all_entities": ["products", "list"], "reasoning": "Specific view type requested", "confidence": 0.9}
    - "what's the weather" -> {"type": "unknown", "entity": "", "all_entities": [], "reasoning": "No Odoo entities found", "confidence": 0.1}
    - "dashboard with sales metrics" -> {"type": "navigation", "entity": "dashboard/sales", "all_entities": ["dashboard", "sales"], "reasoning": "Dashboard with sales context", "confidence": 0.8}
    """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User input: {user_input}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content
            
            result = json.loads(json_str)
            return result
            
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing Groq response: {e}")
            return None
    
    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """Fallback rule-based parsing"""
        # Navigation patterns
        navigation_patterns = [
            r"(?:go to|open|navigate to|show|get to|take me to|bring up) (?:the )?(.*?) (?:page|module|menu|screen)",
            r"(?:go to|open|navigate to|show|get to|take me to|bring up) (?:the )?(.*?)(?:$| page| module| screen)",
            r"(?:get|take) me to (.*?)(?:$| page| module| screen)",
        ]
        
        entity = ""
        
        # Try navigation patterns
        for pattern in navigation_patterns:
            match = re.search(pattern, user_input)
            if match:
                entity = match.group(1).strip()
                break
        
        # Check for single word navigation
        if not entity:
            for key in self.navigation_mapping:
                if key in user_input:
                    entity = key
                    break
        
        if entity:
            return {
                "type": "navigation",
                "entity": entity,
                "entity_model": self._map_to_odoo_model(entity),
                "navigation_targets": self._map_to_navigation_targets(entity),
                "original_input": user_input
            }
        
        return {
            "type": "unknown",
            "entity": "",
            "entity_model": None,
            "navigation_targets": [],
            "original_input": user_input
        }
    
    def _map_to_odoo_model(self, entity: str) -> Optional[str]:
        """Map entity to Odoo model"""
        return self.module_mapping.get(entity)
    
    def _map_to_navigation_targets(self, entity: str) -> List[str]:
        """Map entity to navigation targets"""
        mapping = self.navigation_mapping.get(entity, [])
        return mapping if isinstance(mapping, list) else [mapping] if mapping else []

# ====== IMPROVED NAVIGATION HANDLER ======
class NavigationHandler:
    def __init__(self, odoo_connector: OdooConnector):
        """Initialize with OdooConnector"""
        self.odoo = odoo_connector
    
    def generate_navigation_url(self, parsed_intent: Dict[str, Any]) -> str:
        """Generate Odoo navigation URL based on parsed intent with improved logic"""
        base_url = self.odoo.get_base_url()
        company_id = self.odoo.get_company_id()
        
        # Try each navigation target
        navigation_targets = parsed_intent.get("navigation_targets", [])
        
        for target in navigation_targets:
            menu = self.odoo.get_menu_with_action_details(target)
            if menu:
                url = self._build_url_from_menu(menu, base_url, company_id)
                if url:
                    return url
        
        # Try to find action by model
        if parsed_intent.get("entity_model"):
            action = self.odoo.get_action_by_model(parsed_intent["entity_model"])
            if action:
                url = self._build_url_from_action(action, base_url, company_id)
                if url:
                    return url
        
        # Enhanced fallback URL construction
        entity = parsed_intent.get("entity", "").lower()
        if entity:
            # Try to find menu by entity name directly
            menu = self.odoo.get_menu_with_action_details(entity)
            if menu:
                url = self._build_url_from_menu(menu, base_url, company_id)
                if url:
                    return url
        
        # Final fallback to web interface
        return f"{base_url}/web"
    
    def _build_url_from_menu(self, menu: Dict[str, Any], base_url: str, company_id: int) -> Optional[str]:
        """Build URL from menu information"""
        if not menu.get('action_id'):
            return None
        
        action_id = menu['action_id']
        menu_id = menu['id']
        action_model = menu.get('action_model', '')
        
        # Handle different action types
        if action_model == 'ir.actions.client':
            # For client actions (like dashboards)
            action_details = menu.get('action_details', {})
            if action_details.get('tag') == 'board':
                # Dashboard action
                dashboard_id = self._extract_dashboard_id(action_details)
                if dashboard_id:
                    return f"{base_url}/web#dashboard_id={dashboard_id}&cids={company_id}&menu_id={menu_id}&action={action_id}"
            
            # Generic client action
            return f"{base_url}/web#action={action_id}&menu_id={menu_id}&cids={company_id}"
        
        elif action_model == 'ir.actions.act_window':
            # For window actions
            action_details = menu.get('action_details', {})
            if action_details:
                model = action_details.get('res_model', '')
                view_mode = action_details.get('view_mode', 'list')
                
                # Determine the best view type
                view_type = self._determine_view_type(view_mode, model)
                
                if model:
                    return f"{base_url}/web#action={action_id}&model={model}&view_type={view_type}&cids={company_id}&menu_id={menu_id}"
                else:
                    return f"{base_url}/web#action={action_id}&menu_id={menu_id}&cids={company_id}"
        
        # Generic fallback
        return f"{base_url}/web#action={action_id}&menu_id={menu_id}&cids={company_id}"
    
    def _build_url_from_action(self, action: Dict[str, Any], base_url: str, company_id: int) -> Optional[str]:
        """Build URL from action information"""
        action_id = action['id']
        model = action.get('res_model', '')
        view_mode = action.get('view_mode', 'list')
        
        view_type = self._determine_view_type(view_mode, model)
        
        if model:
            return f"{base_url}/web#action={action_id}&model={model}&view_type={view_type}&cids={company_id}"
        else:
            return f"{base_url}/web#action={action_id}&cids={company_id}"
    
    def _determine_view_type(self, view_mode: str, model: str = '') -> str:
        """Determine the best view type based on view_mode and model"""
        if not view_mode:
            return 'list'
        
        # Split view_mode and get the first one
        view_modes = view_mode.split(',')
        first_view = view_modes[0].strip()
        
        # Model-specific view type preferences
        if model == 'product.template' and 'kanban' in view_modes:
            return 'kanban'
        elif model in ['sale.order', 'account.move', 'purchase.order'] and 'list' in view_modes:
            return 'list'
        elif first_view in ['tree', 'list']:
            return 'list'
        elif first_view == 'form':
            return 'form'
        elif first_view == 'kanban':
            return 'kanban'
        else:
            return first_view
    
    def _extract_dashboard_id(self, action_details: Dict[str, Any]) -> Optional[int]:
        """Extract dashboard ID from action details"""
        try:
            context = action_details.get('context', {})
            if isinstance(context, str):
                context = eval(context)  # Safely evaluate context string
            
            dashboard_id = context.get('dashboard_id')
            if dashboard_id:
                return dashboard_id
            
            # Try to extract from params
            params = action_details.get('params', {})
            if isinstance(params, str):
                params = eval(params)
            
            return params.get('dashboard_id', 2)  # Default dashboard ID
            
        except:
            return 2  # Default dashboard ID

# ====== LANGGRAPH AGENT NODES ======
class OdooNavigationAgent:
    def __init__(self, odoo_connector: OdooConnector, intent_parser: EnhancedIntentParser):
        """Initialize the agent with required components"""
        self.odoo = odoo_connector
        self.intent_parser = intent_parser
        self.navigation_handler = NavigationHandler(odoo_connector)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_intent", self._parse_intent_node)
        workflow.add_node("generate_navigation", self._generate_navigation_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Add edges
        workflow.set_entry_point("parse_intent")
        workflow.add_conditional_edges(
            "parse_intent",
            self._should_navigate,
            {
                "navigate": "generate_navigation",
                "error": "generate_response"
            }
        )
        workflow.add_edge("generate_navigation", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _parse_intent_node(self, state: AgentState) -> AgentState:
        """Parse user intent using enhanced parser"""
        try:
            parsed_intent = self.intent_parser.parse(state["user_input"])
            state["parsed_intent"] = parsed_intent
            state["status"] = "intent_parsed"
            return state
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Error parsing intent: {str(e)}"
            return state
    
    def _generate_navigation_node(self, state: AgentState) -> AgentState:
        """Generate navigation URL"""
        try:
            url = self.navigation_handler.generate_navigation_url(state["parsed_intent"])
            state["odoo_url"] = url
            state["status"] = "navigation_generated"
            return state
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Error generating navigation: {str(e)}"
            return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response"""
        if state["status"] == "error":
            state["response"] = f"❌ Error: {state.get('error_message', 'Unknown error')}"
        elif state["status"] == "navigation_generated":
            entity = state["parsed_intent"].get("entity", "page")
            state["response"] = f"🔗 Navigate to {entity}: {state['odoo_url']}"
        elif state["parsed_intent"].get("type") == "unknown":
            state["response"] = "❓ I can only help with Odoo page navigation. Please specify which Odoo page or module you'd like to navigate to (e.g., 'go to sales', 'open invoices')."
        else:
            state["response"] = "❓ I couldn't understand your navigation request. Please try again."
        
        return state
    
    def _should_navigate(self, state: AgentState) -> str:
        """Determine if we should proceed with navigation"""
        if state["status"] == "error":
            return "error"
        elif state["parsed_intent"].get("type") == "navigation":
            return "navigate"
        else:
            return "error"
    
    def process_input(self, user_input: str) -> str:
        """Process user input and return response"""
        initial_state = AgentState(
            user_input=user_input,
            parsed_intent={},
            odoo_url="",
            status="",
            error_message="",
            response=""
        )
        
        result = self.graph.invoke(initial_state)
        return result["response"]

# ====== MAIN APPLICATION ======
def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Get Groq API key from .env file, command line, or environment variable
    groq_api_key = args.groq_api_key or os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ Groq API key is required.")
        print("Please either:")
        print("  1. Set GROQ_API_KEY in your .env file")
        print("  2. Set GROQ_API_KEY environment variable")
        print("  3. Use --groq-api-key command line argument")
        sys.exit(1)
    
    # Get Odoo password if not provided
    password = args.password
    if not password:
        password = getpass.getpass(f"Enter password for {args.user}: ")
    
    try:
        # Initialize components
        print("🔌 Connecting to Odoo...")
        odoo_connector = OdooConnector(args.host, args.port, args.db, args.user, password)
        
        print("🤖 Initializing AI agent...")
        intent_parser = EnhancedIntentParser(groq_api_key)
        agent = OdooNavigationAgent(odoo_connector, intent_parser)
        
        print("✅ Odoo Navigation AI Agent ready!")
        print("Type 'help' for assistance, 'quit' or 'exit' to quit.")
        print("-" * 50)
        
        # Handle single command mode
        if args.command:
            response = agent.process_input(args.command)
            print(response)
            return
        
        # Interactive mode
        while True:
            try:
                user_input = input("\n🔍 Enter navigation command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("""
🤖 Odoo Navigation AI Agent Help

This agent helps you navigate to different pages in Odoo using natural language.

Examples:
- "go to sales"
- "open invoices" or "open invoicing"
- "take me to customers" or "take me to contacts"
- "show dashboard" or "show dashboards"
- "navigate to products"

Available modules:
sales, invoices, invoicing, customers, contacts, products, inventory, purchases, 
employees, settings, apps, dashboard, dashboards, calendar, discuss, inbox, 
crm, leads, opportunities, website, pos, manufacturing, projects
                    """)
                    continue
                
                # Process the input
                response = agent.process_input(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()