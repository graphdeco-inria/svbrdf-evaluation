import torch

def dot(x, y, dim_):
    return torch.sum(x * y, dim=dim_, keepdim=True)

def length(x, dim = -1, eps = 1e-8):
	# Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    assert x.dtype == torch.float32
    length_ = torch.clamp(dot(x, x, dim), min=eps)
    assert not torch.any(length_ < 1e-8)
    return torch.sqrt(length_.abs() + 1e-8)

def safe_normalize(x, dim = -1, eps = 1e-8):
    assert x.shape[dim] == 3
    return x / length(x, dim, eps)

def normal_from_slopes(normals_xy, dim=-1):
    assert normals_xy.shape[dim] == 2
    norm = torch.clamp(length(normals_xy.float(), dim=dim), min = 0.999)
    normals_xy = 0.999 * normals_xy / norm
    squared_xy = torch.square(normals_xy)
    if dim==-1:
        z_vec = torch.sqrt(torch.clamp(
            1 - squared_xy[..., 0:1] - squared_xy[..., 1:2], 
            min=0.001))
    elif dim==1:
        z_vec = torch.sqrt(torch.clamp(
            1 - squared_xy[:, 0:1, ...] - squared_xy[:, 1:2, ...], 
            min=0.001))
    elif dim==0:
        z_vec = torch.sqrt(torch.clamp(
            1 - squared_xy[0:1, ...] - squared_xy[1:2, ...], 
            min=0.001))
    else:
        print("error in normal_from_slopes! dim=", dim, "unhandled")
    normal = torch.cat((normals_xy, z_vec), dim=dim)
    return normal

## Building an Orthonormal Basis, Revisited
def branchlessONB(n):
	sign = torch.sign(n[:,2])
	a = -1.0 / (sign + n[:,2])
	b = n[:,0] * n[:,1] * a
	b1 = torch.cat([
		1.0 + sign * n[:,0] * n[:,0] * a, 
		sign * b, -sign * n[:,0]], dim=1)
	b2 = torch.cat([
		b, sign + n[:,1] * n[:,1] * a, 
		-n[:,1]], dim=1)
	return b1, b2

def reinhardTonemapper(t):
	return t / (1 + t)

def neuMIPTonemapper(t):
	return torch.log(t + 1)

# from https://github.com/tizian/tonemapper/blob/master/src/operators/HableFilmicOperator.cpp
def unchartedTonemapper(color, exposure = 1.0):
    gamma = 2.2
    A     = 0.15 
    B     = 0.5 
    C     = 0.1 
    D     = 0.2 
    E     = 0.02 
    F     = 0.3 
    W     = 11.2

    def curve(x):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    # Fetch color
    Cin = exposure * color

    # Apply curve directly on color input
    exposureBias = 2.0
    Cout = exposureBias * curve(Cin) / curve(W)

    # Apply gamma curve and clamp
    Cout = torch.pow(Cout, 1.0 / gamma)
    return torch.clamp(Cout, 0.0, 1.0)

def clipTonemapper(t):
	return torch.clamp(t, min=0.0, max=1.0)

def srgb2linrgb(input_color):
    limit = 0.04045
    transformed_color = torch.where(
        input_color > limit,
        torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
        input_color / 12.92
    )  # clamp to stabilize training
    return transformed_color

def gammaCorrection(input):
	"""linrgb2srgb"""
	limit = 0.0031308
	return torch.where(
		input > limit,
		1.055 * torch.pow(
			torch.clamp(
				input, min=limit), 
				(1.0 / 2.4)) - 0.055,
		12.92 * input)

def DeschaintrelogTensor(in_tensor):
	log_001 = torch.log(0.01)
	div_log = torch.log(1.01)-log_001
	return torch.log(in_tensor.add(0.01)).add(-log_001).div(div_log)

def process_raw_render(render):
	render = reinhardTonemapper(render)
	render = gammaCorrection(render)
	return render

def process_render_for_loss(render):
	render = neuMIPTonemapper(render)
	render = gammaCorrection(render)
	return render
