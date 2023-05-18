from setuptools import find_packages, setup


setup(
	name='open_clip_jax',
	version='0.0.1',
	description='Implementation of CLIP in JAX/Flax',
	author='Borna Ahmadzadeh',
	author_email='borna.ahz@gmail.com',
	url='https://github.com/bobmcdear/open-clip-jax',
	packages=find_packages(include='open_clip_jax'),
	install_requires=[
        'flax==0.6.1',
		'huggingface_hub==0.11.1',
		'numpy==1.22.0',
		'optax==0.1.3',
		'pandas==2.0.1',
		'tensorflow==2.11.1',
		'tensorflow_text==2.11.0',
		],
	python_requires='>=3.8',
	)
