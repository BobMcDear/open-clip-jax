from setuptools import find_packages, setup


setup(
	name='open_clip_jax',
	version='0.0.2',
	description='Implementation of CLIP in JAX/Flax',
	author='Borna Ahmadzadeh',
	author_email='borna.ahz@gmail.com',
	url='https://github.com/bobmcdear/open-clip-jax',
	packages=find_packages(),
	install_requires=[
        'flax<=0.8.0',
		'huggingface_hub',
		'optax',
		'tensorflow',
		'tensorflow_text',
		],
    include_package_data=True,
	python_requires='>=3.8',
	)
