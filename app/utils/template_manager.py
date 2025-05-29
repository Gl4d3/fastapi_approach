import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FieldRegion:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class FieldConfig:
    name: str
    display_name: str
    region: FieldRegion
    ocr_config: str
    preprocessing: Optional[str] = None
    validation_pattern: Optional[str] = None
    required: bool = True
    data_type: str = "string"
    post_processing: Optional[str] = None

@dataclass
class DocumentTemplate:
    document_type: str
    display_name: str
    description: str
    version: str
    fields: List[FieldConfig]
    classification_rules: Dict[str, Any]
    validation_rules: Dict[str, Any]
    created_by: str
    created_at: str
    updated_at: str

class TemplateManager:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates: Dict[str, DocumentTemplate] = {}
    
    def load_templates(self):
        """Load all templates"""
        # Create default templates if directory is empty
        if not any(self.templates_dir.glob("*.yaml")):
            self._create_default_templates()
        
        # Load existing templates
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    template = self._dict_to_template(template_data)
                    self.templates[template.document_type] = template
                    logger.info(f"Loaded template: {template.document_type}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
    
    def _dict_to_template(self, data: Dict) -> DocumentTemplate:
        """Convert dictionary to DocumentTemplate"""
        fields = []
        for field_data in data['fields']:
            region = FieldRegion(**field_data['region'])
            field = FieldConfig(
                name=field_data['name'],
                display_name=field_data['display_name'],
                region=region,
                ocr_config=field_data['ocr_config'],
                preprocessing=field_data.get('preprocessing'),
                validation_pattern=field_data.get('validation_pattern'),
                required=field_data.get('required', True),
                data_type=field_data.get('data_type', 'string'),
                post_processing=field_data.get('post_processing')
            )
            fields.append(field)
        
        return DocumentTemplate(
            document_type=data['document_type'],
            display_name=data['display_name'],
            description=data['description'],
            version=data['version'],
            fields=fields,
            classification_rules=data['classification_rules'],
            validation_rules=data['validation_rules'],
            created_by=data['created_by'],
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )
    
    def get_template(self, document_type: str) -> Optional[DocumentTemplate]:
        """Get template by type"""
        return self.templates.get(document_type)
    
    def _create_default_templates(self):
        """Create default templates"""
        # KRA PIN Template
        kra_template = DocumentTemplate(
            document_type="kra_pin",
            display_name="KRA PIN Certificate",
            description="Kenya Revenue Authority Personal Identification Number Certificate",
            version="1.0.0",
            fields=[
                FieldConfig(
                    name="pin_number",
                    display_name="PIN Number",
                    region=FieldRegion(0.65, 0.15, 0.95, 0.25),
                    ocr_config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    validation_pattern=r"A\d{9}[A-Z]",
                    preprocessing="enhance"
                ),
                FieldConfig(
                    name="taxpayer_name",
                    display_name="Taxpayer Name",
                    region=FieldRegion(0.45, 0.35, 0.95, 0.45),
                    ocr_config="--psm 6",
                    validation_pattern=r"[A-Z][A-Z\s]+",
                    preprocessing="table_cell",
                    post_processing="clean_name"
                ),
                FieldConfig(
                    name="email_address",
                    display_name="Email Address",
                    region=FieldRegion(0.45, 0.45, 0.95, 0.55),
                    ocr_config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.",
                    validation_pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    preprocessing="table_cell",
                    post_processing="extract_email",
                    required=False
                )
            ],
            classification_rules={
                "text_patterns": ["PIN Certificate", "Kenya Revenue Authority", "A\\d{9}[A-Z]"],
                "layout_features": {"has_table": True}
            },
            validation_rules={},
            created_by="system",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Save default template
        self._save_template_to_file(kra_template)
        
        # Create similar templates for other document types...
        # (Kenyan ID, Business Cert, etc.)
    
    def _save_template_to_file(self, template: DocumentTemplate):
        """Save template to YAML file"""
        template_file = self.templates_dir / f"{template.document_type}.yaml"
        template_dict = asdict(template)
        
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template_dict, f, default_flow_style=False, allow_unicode=True)