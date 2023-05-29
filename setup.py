from setuptools import find_packages, setup


setup(
	name='open_clip_jax',
	version='0.0.1',
	description='Implementation of CLIP in JAX/Flax',
	author='Borna Ahmadzadeh',
	author_email='borna.ahz@gmail.com',
	url='https://github.com/bobmcdear/open-clip-jax',
	packages=find_packages(),
	install_requires=[
        'flax',
		'huggingface_hub',
		'optax',
		'pandas',
		'tensorflow',
		'tensorflow_text',
		],
    include_package_data=True,
	python_requires='>=3.8',
	)
