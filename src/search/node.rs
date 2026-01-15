pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
}

pub struct Root;
impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
}

pub struct PV;
impl NodeType for PV {
    const PV: bool = true;
    const ROOT: bool = false;
}

pub struct NonPV;
impl NodeType for NonPV {
    const PV: bool = false;
    const ROOT: bool = false;
}

